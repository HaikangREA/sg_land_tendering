import pandas as pd
import numpy as np
from glspred import preprocess

pred_raw = pd.read_csv(r'G:\REA\Working files\land-bidding\pipeline\pred.csv')
master = pd.read_csv(r'G:\REA\Working files\land-bidding\pipeline\gls_origin.csv')
pred = pred_raw.copy()
processor = preprocess.Preprocess()
pred_processed = processor.process(pred, master)


check = 42