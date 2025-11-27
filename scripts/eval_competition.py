import re
import time
import pickle
from pathlib import Path
import numpy as np

import torch
import typer
from tqdm import tqdm
from neural_decoder.dataset import SpeechDataset
from neural_decoder.neural_decoder_trainer import getDatasetLoaders
from neural_decoder.neural_decoder_trainer import loadModel
import neural_decoder.utils.lmDecoderUtils as lmDecoderUtils

app = typer.Typer()


@app.command()
def main(
    model_path: str = typer.Option(..., help="Path to trained model directory"),
    dataset_path: str = typer.Option(
        None,
        help="Path to dataset directory (default: ./data/ptDecoder_ctc relative to repo root)",
    ),
    partition: str = typer.Option(
        "competition", help="Partition to evaluate: 'competition' or 'test'"
    ),
    acoustic_scale: float = typer.Option(0.5, help="Acoustic scale for LM decoding"),
    blank_penalty: float = typer.Option(
        np.log(7), help="Blank penalty for CTC decoding"
    ),
    llm_weight: float = typer.Option(0.5, help="Weight for LLM rescoring"),
):
    """Evaluate competition data with language model decoding."""

    print(f"Loading model from {model_path}")
    with open(model_path + "/args", "rb") as handle:
        args = pickle.load(handle)

    # Set dataset path: use provided path or default to ./data/ptDecoder_ctc relative to repo root
    if dataset_path is None:
        repo_root = Path(__file__).parent.parent
        dataset_path = str(repo_root / "data" / "ptDecoder_ctc")

    args["datasetPath"] = dataset_path
    print(f"Using dataset from {dataset_path}")
    trainLoaders, testLoaders, loadedData = getDatasetLoaders(
        args["datasetPath"], args["seqLen"], args["maxTimeSeriesLen"], args["batchSize"]
    )

    model = loadModel(model_path, device="cpu")
    device = "cpu"
    model.eval()

    rnn_outputs = {
        "logits": [],
        "logitLengths": [],
        "trueSeqs": [],
        "transcriptions": [],
    }

    if partition == "competition":
        testDayIdxs = [4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20]
    elif partition == "test":
        testDayIdxs = range(len(loadedData[partition]))
    else:
        raise ValueError(
            f"Invalid partition: {partition}. Must be 'competition' or 'test'"
        )

    print("Generating RNN outputs...")
    for i, testDayIdx in tqdm(
        enumerate(testDayIdxs), total=len(testDayIdxs), desc="Processing days"
    ):
        test_ds = SpeechDataset([loadedData[partition][i]])
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=1, shuffle=False, num_workers=0
        )
        for j, (X, y, X_len, y_len, _) in enumerate(test_loader):
            X, y, X_len, y_len, dayIdx = (
                X.to(device),
                y.to(device),
                X_len.to(device),
                y_len.to(device),
                torch.tensor([testDayIdx], dtype=torch.int64).to(device),
            )
            pred = model.forward(X, dayIdx)
            adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)

            for iterIdx in range(pred.shape[0]):
                trueSeq = np.array(y[iterIdx][0 : y_len[iterIdx]].cpu().detach())

                rnn_outputs["logits"].append(pred[iterIdx].cpu().detach().numpy())
                rnn_outputs["logitLengths"].append(
                    adjustedLens[iterIdx].cpu().detach().item()
                )
                rnn_outputs["trueSeqs"].append(trueSeq)

            transcript = loadedData[partition][i]["transcriptions"][j].strip()
            transcript = re.sub(r"[^a-zA-Z\- \']", "", transcript)
            transcript = transcript.replace("--", "").lower()
            rnn_outputs["transcriptions"].append(transcript)

    print("Loading language models...")
    MODEL_CACHE_DIR = "/scratch/users/stfan/huggingface"
    # Load OPT 6B model
    llm, llm_tokenizer = lmDecoderUtils.build_opt(
        cacheDir=MODEL_CACHE_DIR, device="auto", load_in_8bit=True
    )

    lmDir = "/oak/stanford/groups/henderj/stfan/code/nptlrig2/LanguageModelDecoder/examples/speech/s0/lm_order_exp/5gram/data/lang_test"
    ngramDecoder = lmDecoderUtils.build_lm_decoder(
        lmDir, acoustic_scale=acoustic_scale, nbest=100, beam=18
    )

    # Generate nbest outputs from 5gram LM
    print("Decoding with 5-gram LM...")
    start_t = time.time()
    nbest_outputs = []
    for j in tqdm(range(len(rnn_outputs["logits"])), desc="5-gram decoding"):
        logits = rnn_outputs["logits"][j]
        logits = np.concatenate(
            [logits[:, 1:], logits[:, 0:1]], axis=-1
        )  # Blank is last token
        logits = lmDecoderUtils.rearrange_speech_logits(
            logits[None, :, :], has_sil=True
        )
        nbest = lmDecoderUtils.lm_decode(
            ngramDecoder,
            logits[0],
            blankPenalty=blank_penalty,
            returnNBest=True,
            rescore=True,
        )
        nbest_outputs.append(nbest)
    time_per_sample = (time.time() - start_t) / len(rnn_outputs["logits"])
    print(f"5-gram decoding took {time_per_sample:.3f} seconds per sample")

    print("Converting transcriptions...")
    for i in tqdm(
        range(len(rnn_outputs["transcriptions"])), desc="Converting", leave=False
    ):
        new_trans = [ord(c) for c in rnn_outputs["transcriptions"][i]] + [0]
        rnn_outputs["transcriptions"][i] = np.array(new_trans)

    # Rescore nbest outputs with LLM
    print("Rescoring with LLM...")
    start_t = time.time()
    llm_out = lmDecoderUtils.cer_with_gpt2_decoder(
        llm,
        llm_tokenizer,
        nbest_outputs[:],
        acoustic_scale,
        rnn_outputs,
        outputType="speech_sil",
        returnCI=True,
        lengthPenalty=0,
        alpha=llm_weight,
    )
    time_per_sample = (time.time() - start_t) / len(rnn_outputs["logits"])
    print(f"LLM decoding took {time_per_sample:.3f} seconds per sample")

    print(f"\nResults: CER={llm_out['cer']:.4f}, WER={llm_out['wer']:.4f}")

    print(f"Saving results to {model_path}")
    with open(model_path + "/llm_out", "wb") as handle:
        pickle.dump(llm_out, handle)

    decodedTranscriptions = llm_out["decoded_transcripts"]
    output_file = model_path + "/5gramLLMCompetitionSubmission.txt"
    with open(output_file, "w") as f:
        for x in range(len(decodedTranscriptions)):
            f.write(decodedTranscriptions[x] + "\n")

    print(f"Submission file saved to {output_file}")


if __name__ == "__main__":
    app()
