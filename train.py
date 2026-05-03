import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from collections import Counter
import tqdm

from model import SeverityClassifier

# fixed seed
torch.manual_seed(42)
np.random.seed(42)


def computeECE(probs, labels, nBins=15):
    # expected calibration error
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels).astype(float)
    ece = 0.0
    for i in range(nBins):
        lo = i / nBins
        hi = (i + 1) / nBins
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        binAcc  = correct[mask].mean()
        binConf = confidences[mask].mean()
        ece += mask.mean() * abs(binAcc - binConf)
    return ece


def collate(batch, tokenizer, maxLen=128):
    images  = torch.stack([b["image"] for b in batch]).float()
    labels  = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    texts   = [b["clinicalText"] for b in batch]
    enc     = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=maxLen)
    return images, labels, enc["input_ids"], enc["attention_mask"]


def evaluate(model, loader, device, split="val"):
    model.eval()
    allLogits, allLabels = [], []
    with torch.no_grad():
        for images, labels, inputIds, attnMask in loader:
            images   = images.to(device)
            inputIds = inputIds.to(device)
            attnMask = attnMask.to(device)
            logits, _ = model(images, inputIds, attnMask)
            allLogits.append(logits.cpu())
            allLabels.append(labels)

    allLogits = torch.cat(allLogits)
    allLabels = torch.cat(allLabels).numpy()
    allProbs  = torch.softmax(allLogits, dim=-1).numpy()
    allPreds  = allProbs.argmax(axis=1)

    acc   = (allPreds == allLabels).mean()
    f1    = f1_score(allLabels, allPreds, average="macro")
    # AUROC: one-vs-rest
    labBin = label_binarize(allLabels, classes=[0, 1, 2])
    try:
        auroc = roc_auc_score(labBin, allProbs, multi_class="ovr", average="macro")
    except ValueError:
        auroc = float("nan")
    ece   = computeECE(allProbs, allLabels)

    print("[%s] acc=%.4f  f1=%.4f  auroc=%.4f  ece=%.4f" % (split, acc, f1, auroc, ece))
    return acc, f1, auroc, ece


def main(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # import dataset
    import sys
    sys.path.insert(0, args.projectDir)
    from dataset import GliomaSeverityDataset

    trainSet = GliomaSeverityDataset(args.manifest, split="TRAIN")
    testSet  = GliomaSeverityDataset(args.manifest, split="TEST")

    collateFn = lambda b: collate(b, tokenizer)
    trainLoader = DataLoader(trainSet, batch_size=args.batchSize, shuffle=True,
                             num_workers=1, collate_fn=collateFn, pin_memory=True)
    testLoader  = DataLoader(testSet,  batch_size=args.batchSize, shuffle=False,
                             num_workers=1, collate_fn=collateFn, pin_memory=True)

    # class weights for imbalance (computed from patient-level counts: low=76, mid=70, high=140)
    # use inverse frequency at slice level
    labelCounts = Counter(int(s["severity"]) for s in trainSet.samples)
    total = sum(labelCounts.values())
    weights = torch.tensor(
        [total / (3 * labelCounts[c]) for c in range(3)], dtype=torch.float
    ).to(device)
    print("class weights: %s" % str(weights.cpu().numpy()))

    model     = SeverityClassifier(dropoutRate=args.dropoutRate).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.outputDir, exist_ok=True)
    bestF1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        totalLoss = 0.0
        correct   = 0
        total     = 0

        pbar = tqdm.tqdm(trainLoader, desc="epoch %d" % epoch)
        for images, labels, inputIds, attnMask in pbar:
            images   = images.to(device)
            labels   = labels.to(device)
            inputIds = inputIds.to(device)
            attnMask = attnMask.to(device)

            optimizer.zero_grad()
            logits, _ = model(images, inputIds, attnMask)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            totalLoss += loss.item() * labels.size(0)
            correct   += (logits.argmax(1) == labels).sum().item()
            total     += labels.size(0)
            pbar.set_postfix({"loss": "%.4f" % (totalLoss / total),
                              "acc":  "%.4f" % (correct / total)})

        scheduler.step()

        acc, f1, auroc, ece = evaluate(model, testLoader, device, split="test")

        # save best model by macro F1
        if f1 > bestF1:
            bestF1 = f1
            torch.save(model.state_dict(), os.path.join(args.outputDir, "best_model.pt"))
            print("  saved best model (f1=%.4f)" % bestF1)

    print("\nTraining complete. Best macro F1: %.4f" % bestF1)

    # final evaluation with MC dropout uncertainty
    print("\nRunning MC dropout uncertainty on test set...")
    model.load_state_dict(torch.load(os.path.join(args.outputDir, "best_model.pt")))
    allMeanProbs, allUnc, allLabels = [], [], []
    for images, labels, inputIds, attnMask in testLoader:
        images   = images.to(device)
        inputIds = inputIds.to(device)
        attnMask = attnMask.to(device)
        meanProbs, unc = model.predictWithUncertainty(images, inputIds, attnMask, nSamples=20)
        allMeanProbs.append(meanProbs.cpu())
        allUnc.append(unc.cpu())
        allLabels.append(labels)

    allMeanProbs = torch.cat(allMeanProbs).numpy()
    allUnc       = torch.cat(allUnc).numpy()
    allLabels    = torch.cat(allLabels).numpy()
    allPreds     = allMeanProbs.argmax(axis=1)

    acc  = (allPreds == allLabels).mean()
    f1   = f1_score(allLabels, allPreds, average="macro")
    ece  = computeECE(allMeanProbs, allLabels)
    print("[MC test] acc=%.4f  f1=%.4f  ece=%.4f" % (acc, f1, ece))
    print("[MC test] mean uncertainty=%.6f  std=%.6f" % (allUnc.mean(), allUnc.std()))

    # save results
    resultsPath = os.path.join(args.outputDir, "results.csv")
    with open(resultsPath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subjectId", "sliceIdx", "trueLabel", "predLabel",
                    "prob0", "prob1", "prob2", "uncertainty"])
        for i, s in enumerate(testSet.samples):
            w.writerow([s["subjectId"], s["sliceIdx"], allLabels[i], allPreds[i],
                        "%.6f" % allMeanProbs[i, 0],
                        "%.6f" % allMeanProbs[i, 1],
                        "%.6f" % allMeanProbs[i, 2],
                        "%.6f" % allUnc[i]])
    print("results saved to %s" % resultsPath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",    default="/home/ob675831/tumor_progression/processed/manifest_clean.csv")
    parser.add_argument("--projectDir",  default="/home/ob675831/tumor_progression")
    parser.add_argument("--outputDir",   default="/home/ob675831/tumor_progression/output")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batchSize",   type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--dropoutRate", type=float, default=0.3)
    args = parser.parse_args()
    main(args)
