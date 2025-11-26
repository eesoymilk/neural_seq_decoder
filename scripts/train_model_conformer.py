import os
import pickle
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from edit_distance import SequenceMatcher

# Import the new model
from neural_decoder.conformer_model import ConformerDecoder
from neural_decoder.dataset import SpeechDataset

def getDatasetLoaders(datasetName: str, batchSize: int):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def trainModel(cfg: DictConfig):
    # Convert OmegaConf object to primitive dict for pickling if needed, 
    # but we can access it directly.
    # Also ensure outputDir exists
    os.makedirs(cfg.outputDir, exist_ok=True)
    
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Save args
    with open(os.path.join(cfg.outputDir, "args"), "wb") as file:
        pickle.dump(OmegaConf.to_container(cfg, resolve=True), file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        cfg.datasetPath,
        cfg.batchSize,
    )

    # Initialize Conformer Model
    model = ConformerDecoder(
        neural_dim=cfg.nInputFeatures,
        n_classes=cfg.nClasses,
        hidden_dim=cfg.nUnits,
        layer_dim=cfg.nLayers,
        nDays=len(loadedData["train"]),
        dropout=cfg.dropout,
        device=device,
        strideLen=cfg.strideLen,
        kernelLen=cfg.kernelLen,
        gaussianSmoothWidth=cfg.gaussianSmoothWidth,
        bidirectional=cfg.bidirectional,
    ).to(device)

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    
    # Using AdamW as suggested in project.txt for Transformers/Conformers
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lrStart,
        betas=(0.9, 0.999),
        eps=1e-8, # Standard epsilon
        weight_decay=cfg.l2_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=cfg.lrEnd / cfg.lrStart,
        total_iters=cfg.nBatch,
    )

    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    
    # Iterator for training
    train_iterator = iter(trainLoader)
    
    for batch in range(cfg.nBatch):
        model.train()

        try:
            X, y, X_len, y_len, dayIdx = next(train_iterator)
        except StopIteration:
            train_iterator = iter(trainLoader)
            X, y, X_len, y_len, dayIdx = next(train_iterator)

        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Noise augmentation is faster on GPU
        if cfg.whiteNoiseSD > 0:
            X += torch.randn(X.shape, device=device) * cfg.whiteNoiseSD

        if cfg.constantOffsetSD > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * cfg.constantOffsetSD
            )

        # Compute prediction error
        pred = model.forward(X, dayIdx)

        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    pred = model.forward(X, testDayIdx)
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime) / 100:>7.3f}"
                )
                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), os.path.join(cfg.outputDir, "modelWeights"))
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(os.path.join(cfg.outputDir, "trainingStats"), "wb") as file:
                pickle.dump(tStats, file)

@hydra.main(version_base="1.1", config_path="../src/neural_decoder/conf", config_name="conformer")
def main(cfg: DictConfig):
    # Fix dataset path if it's the default one from config which might be wrong for this user
    # The user's dataset path is "/Users/soymilk/Codes/neural_seq_decoder/data/ptDecoder_ctc"
    # The config has "/oak/stanford/groups/henderj/stfan/data/ptDecoder_ctc"
    # We should probably override it if it matches the default, or just let the user override it via CLI.
    # But since I know the correct path from the previous script, I'll set it here if not provided?
    # Actually, better to update the config file or let the user handle it. 
    # But for this "beat the baseline" task, I should make it work out of the box.
    # I will update the conformer.yaml to have the correct local path or override it here.
    # Let's override it in conformer.yaml actually, but I already wrote it.
    # I will just pass it in the command line or update conformer.yaml.
    # Let's update conformer.yaml in the next step if needed, or just hardcode the fix here for now as a default?
    # No, hardcoding in code defeats the purpose of config.
    # I will update conformer.yaml to use the correct path.
    
    trainModel(cfg)

if __name__ == "__main__":
    main()
