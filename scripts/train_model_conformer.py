import os
import pickle
import time
import logging
import hydra
from neural_decoder.squeezeformer_model import SqueezeFormerDecoder
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from edit_distance import SequenceMatcher
from tqdm import tqdm
from neural_decoder.conformer_model import ConformerDecoder
from neural_decoder.dataset import SpeechDataset

log = logging.getLogger(__name__)


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


def move_batch_to_device(X, y, X_len, y_len, dayIdx, device):
    """Move batch data to specified device."""
    return (
        X.to(device),
        y.to(device),
        X_len.to(device),
        y_len.to(device),
        dayIdx.to(device),
    )


def compute_adjusted_lengths(X_len, kernel_len, stride_len):
    """Compute adjusted sequence lengths after convolution."""
    return ((X_len - kernel_len) / stride_len).to(torch.int32)


def compute_ctc_loss(model, loss_ctc, X, y, X_len, y_len, dayIdx):
    """Compute CTC loss and return predictions, loss, and adjusted lengths."""
    pred = model.forward(X, dayIdx)
    adjusted_lens = compute_adjusted_lengths(X_len, model.kernelLen, model.strideLen)
    loss = loss_ctc(
        torch.permute(pred.log_softmax(2), [1, 0, 2]),
        y,
        adjusted_lens,
        y_len,
    )
    return pred, loss, adjusted_lens


def decode_and_compute_cer(pred, y, y_len, adjusted_lens):
    """Decode predictions and compute character error rate metrics."""
    total_edit_distance = 0
    total_seq_length = 0

    for iterIdx in range(pred.shape[0]):
        decodedSeq = torch.argmax(
            pred[iterIdx, 0 : adjusted_lens[iterIdx], :],
            dim=-1,
        )
        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
        decodedSeq = decodedSeq.cpu().detach().numpy()
        decodedSeq = np.array([i for i in decodedSeq if i != 0])

        trueSeq = np.array(y[iterIdx][0 : y_len[iterIdx]].cpu().detach())

        matcher = SequenceMatcher(a=trueSeq.tolist(), b=decodedSeq.tolist())
        total_edit_distance += matcher.distance()
        total_seq_length += len(trueSeq)

    return total_edit_distance, total_seq_length


def evaluate_model(model, testLoader, loss_ctc, device):
    """Evaluate model on test set and return average loss and CER."""
    model.eval()
    allLoss = []
    total_edit_distance = 0
    total_seq_length = 0

    for X, y, X_len, y_len, testDayIdx in tqdm(
        testLoader, desc="Evaluating", leave=False
    ):
        X, y, X_len, y_len, testDayIdx = move_batch_to_device(
            X, y, X_len, y_len, testDayIdx, device
        )
        pred, loss, adjusted_lens = compute_ctc_loss(
            model, loss_ctc, X, y, X_len, y_len, testDayIdx
        )

        allLoss.append(loss.cpu().detach().numpy())
        batch_edit_dist, batch_seq_len = decode_and_compute_cer(
            pred, y, y_len, adjusted_lens
        )
        total_edit_distance += batch_edit_dist
        total_seq_length += batch_seq_len

    avg_loss = np.sum(allLoss) / len(testLoader)
    cer = total_edit_distance / total_seq_length
    return avg_loss, cer


def trainModel(cfg: DictConfig):
    # Convert OmegaConf object to primitive dict for pickling if needed,
    # but we can access it directly.
    # Also ensure outputDir exists
    os.makedirs(cfg.outputDir, exist_ok=True)

    # Create a file-only logger for training progress to avoid disrupting tqdm
    file_logger = logging.getLogger("file_only")
    file_logger.setLevel(logging.INFO)
    file_logger.propagate = False  # Don't propagate to root logger
    file_handler = logging.FileHandler(
        os.path.join(cfg.outputDir, "train_model_conformer.log"), mode="a"
    )
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    )
    file_logger.addHandler(file_handler)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    log.info(f"Using device: {device}")

    # Save args
    with open(os.path.join(cfg.outputDir, "args"), "wb") as file:
        pickle.dump(OmegaConf.to_container(cfg, resolve=True), file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        cfg.datasetPath,
        cfg.batchSize,
    )

    # Initialize Model based on config
    if "squeezeformer" in cfg.modelName.lower():
        print("Initializing SqueezeFormer Model")
        model = SqueezeFormerDecoder(
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
    else:
        print("Initializing Conformer Model")
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
        eps=1e-8,  # Standard epsilon
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

    pbar = tqdm(range(cfg.nBatch), desc="Training", unit="batch")
    for batch in pbar:
        model.train()

        try:
            X, y, X_len, y_len, dayIdx = next(train_iterator)
        except StopIteration:
            train_iterator = iter(trainLoader)
            X, y, X_len, y_len, dayIdx = next(train_iterator)

        X, y, X_len, y_len, dayIdx = move_batch_to_device(
            X, y, X_len, y_len, dayIdx, device
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
        pred, loss, adjusted_lens = compute_ctc_loss(
            model, loss_ctc, X, y, X_len, y_len, dayIdx
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
                avgDayLoss, cer = evaluate_model(model, testLoader, loss_ctc, device)

                endTime = time.time()
                time_per_batch = (endTime - startTime) / 100

                # Update progress bar with metrics
                pbar.set_postfix(
                    {
                        "loss": f"{avgDayLoss:.4f}",
                        "cer": f"{cer:.4f}",
                        "time/batch": f"{time_per_batch:.3f}s",
                    }
                )

                # Log to file only (using file_logger to avoid console output that disrupts tqdm)
                file_logger.info(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {time_per_batch:>7.3f}"
                )

                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(
                    model.state_dict(), os.path.join(cfg.outputDir, "modelWeights")
                )
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(os.path.join(cfg.outputDir, "trainingStats"), "wb") as file:
                pickle.dump(tStats, file)


@hydra.main(
    version_base="1.1",
    config_path="../src/neural_decoder/conf",
    config_name="conformer",
)
def main(cfg: DictConfig):
    trainModel(cfg)


if __name__ == "__main__":
    main()
