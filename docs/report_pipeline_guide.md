# Huong dan chay pipeline bao cao

File script chinh:

```powershell
python scripts\run_report_pipeline.py
```

Script nay gom cac quy trinh chinh de phuc vu bao cao:

- Kiem tra data dau vao.
- Train Hybrid model.
- Train va so sanh Popularity, MF, NCF, Hybrid.
- Test lai cac checkpoint da train.
- Ghi log rieng cho tung stage.
- Ghi metrics CSV va bieu do PNG trong thu muc `reports/`.

## 1. Chay nhanh de kiem tra moi thu co hoat dong

Dung lenh nay truoc khi chay bao cao that:

```powershell
python scripts\run_report_pipeline.py --profile smoke
```

Profile `smoke` chi dung mot phan nho data, it epoch, it user evaluation. Muc dich la kiem tra:

- Data co du file hay khong.
- Code train/test co chay het pipeline hay khong.
- Checkpoint co luu/load duoc khong.
- Log va output co duoc tao dung khong.

Output:

```text
reports/report_pipeline/smoke/
reports/report_pipeline/smoke/logs/
```

## 2. Chay de lay ket qua bao cao

Day la lenh nen dung cho bao cao neu may khong qua manh:

```powershell
python scripts\run_report_pipeline.py --profile report
```

Profile `report` mac dinh:

- Dung 200,000 dong train gan nhat.
- Train Hybrid 3 epochs.
- Train MF/NCF 3 epochs.
- Evaluate 1,000 users.
- Sample 1,000 negative candidates moi user.

Output chinh:

```text
reports/report_pipeline/report/
reports/report_pipeline/report/train_compare/
reports/report_pipeline/report/test_results/
reports/report_pipeline/report/logs/
```

File can dua vao bao cao:

```text
reports/report_pipeline/report/train_compare/metrics_comparison.csv
reports/report_pipeline/report/train_compare/training_history.csv
reports/report_pipeline/report/train_compare/model_metric_comparison.png
reports/report_pipeline/report/train_compare/training_loss_curves.png
reports/report_pipeline/report/test_results/metrics_comparison.csv
reports/report_pipeline/report/test_results/model_metric_comparison.png
```

Checkpoint mac dinh:

```text
checkpoints/hybrid_best.pt
checkpoints/hybrid_last.pt
checkpoints/mf_best.pt
checkpoints/ncf_best.pt
```

## 3. Epoch va train/test la gi

So epoch mac dinh theo tung profile:

```text
profile   Hybrid epochs   MF epochs   NCF epochs   Popularity epochs
smoke     1               1           1            0
report    3               3           3            0
full      5               5           5            0
```

Giai thich:

- `Popularity` khong train bang gradient nen khong co epoch. No chi dem tan suat item trong train set.
- `MF` va `NCF` duoc train trong stage `train-compare`, so epoch lay tu `--compare-epochs`.
- `Hybrid` duoc train rieng trong stage `train-hybrid`, so epoch lay tu `--hybrid-epochs`.
- `full` dat Hybrid 5 epochs de dong bo voi MF/NCF va tranh chay qua lau. Hybrid van co early stopping, nen co the dung som neu validation loss khong cai thien.

Tai sao can nhieu epoch:

- Mot epoch la mot luot model hoc qua tap train hien tai.
- Deep learning can nhieu lan cap nhat trong so, mot epoch thuong chua du de loss on dinh.
- Hybrid hoc nhieu thanh phan hon MF/NCF: user/item embedding, MLP, visual feature, metadata fusion.
- Moi epoch co negative sampling, nen model co co hoi thay cac negative items khac nhau.

Trong pipeline co test lai:

- `train-hybrid`: train Hybrid, validation moi epoch, luu `hybrid_best.pt`, sau do tinh MAP@12 tren test set.
- `train-compare`: train MF/NCF, validation moi epoch, evaluate tren test candidates, ghi metrics va plot so sanh.
- `test`: load lai checkpoint `mf_best.pt`, `ncf_best.pt`, `hybrid_best.pt` va test lai tren test set. Day la stage nen dung de lay bang metric cuoi cung tu checkpoint da luu.

Noi ngan gon: `train-hybrid` va `train-compare` co train; `test` la test lai checkpoint, khong train.

## 4. Chay full dataset

Chi nen chay neu co GPU va du thoi gian:

```powershell
python scripts\run_report_pipeline.py --profile full
```

Profile `full`:

- Dung toan bo train set.
- Train Hybrid 5 epochs.
- Train MF/NCF 5 epochs.
- Evaluate tat ca test users.

Neu may dang chay CPU, lenh nay co the rat lau.

## 5. Khong ghi de checkpoint cu

Neu muon giu checkpoint hien tai trong `checkpoints/`, dung checkpoint dir rieng:

```powershell
python scripts\run_report_pipeline.py --profile report --checkpoint-dir reports\report_pipeline\checkpoints
```

Luc nay checkpoint se nam o:

```text
reports/report_pipeline/checkpoints/
```

## 6. Chay tung stage rieng

Danh sach stage:

```text
check-data
train-hybrid
train-compare
test
```

Chay chi mot stage:

```powershell
python scripts\run_report_pipeline.py --profile report --stages check-data
python scripts\run_report_pipeline.py --profile report --stages train-hybrid
python scripts\run_report_pipeline.py --profile report --stages train-compare
python scripts\run_report_pipeline.py --profile report --stages test
```

Chay nhieu stage:

```powershell
python scripts\run_report_pipeline.py --profile report --stages train-hybrid,train-compare,test
```

Dung `--stages test` khi da co checkpoint va chi muon test lai:

```powershell
python scripts\run_report_pipeline.py --profile report --stages test
```

## 7. Dieu chinh tham so khi may cham

Giam data train:

```powershell
python scripts\run_report_pipeline.py --profile report --max-train-rows 50000
```

Giam epoch:

```powershell
python scripts\run_report_pipeline.py --profile report --hybrid-epochs 1 --compare-epochs 1
```

Giam so user evaluation:

```powershell
python scripts\run_report_pipeline.py --profile report --max-eval-users 200
```

Giam negative candidates:

```powershell
python scripts\run_report_pipeline.py --profile report --negative-candidates 200
```

Tat AMP neu gap loi mixed precision:

```powershell
python scripts\run_report_pipeline.py --profile report --no-amp
```

## 8. Chi chay mot so model

Chi train/test baseline:

```powershell
python scripts\run_report_pipeline.py --profile report --models popularity
```

Chi train/test MF va NCF:

```powershell
python scripts\run_report_pipeline.py --profile report --models mf,ncf
```

Chay day du:

```powershell
python scripts\run_report_pipeline.py --profile report --models popularity,mf,ncf,hybrid
```

## 9. Doc log khi bi treo hoac loi

Log nam trong:

```text
reports/report_pipeline/<profile>/logs/
```

Vi du:

```text
reports/report_pipeline/report/logs/train-hybrid.log
reports/report_pipeline/report/logs/train-compare.log
reports/report_pipeline/report/logs/test.log
```

Neu terminal dung lau, mo log cua stage dang chay de xem dang o batch nao.
Script `train_hybrid.py` da co heartbeat log, nen neu van dang chay se co dong dang:

```text
Still running: epoch ...
```

## 10. Lenh khuyen dung cho bao cao

Thu tu thuc te nen lam:

```powershell
python scripts\run_report_pipeline.py --profile smoke
python scripts\run_report_pipeline.py --profile report
```

Neu `report` qua cham:

```powershell
python scripts\run_report_pipeline.py --profile report --max-train-rows 50000 --max-eval-users 300 --hybrid-epochs 1 --compare-epochs 1
```

Neu can ket qua tot hon va may co GPU:

```powershell
python scripts\run_report_pipeline.py --profile full
```
