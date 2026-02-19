# eSASRec (ML-20M skeleton)

Компактная реализация eSASRec из статьи **"eSASRec: Enhancing Transformer-based Recommendations in a Modular Fashion" (arXiv:2508.06450)**:
- Shifted Sequence objective (как в SASRec)
- LiGR-блоки (pre-norm + gated residual)
- Sampled Softmax loss
- Метрики: Recall@10, NDCG@10 (full-catalog) и apples-to-apples S3Rec-style sampled protocol: HR@{1,5,10}, NDCG@{5,10}, MRR

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Подготовка данных ML-20M

Ожидается `ratings.csv` из MovieLens 20M.

```bash
python -m esasrec.train \
  --ratings-path /path/to/ml-20m/ratings.csv \
  --workdir ./artifacts \
  --device cpu \
  --epochs 1 \
  --max-train-users 20000 \
  --max-eval-users 5000
```

## Полный запуск на GPU

```bash
python -m esasrec.train \
  --ratings-path /path/to/ml-20m/ratings.csv \
  --workdir ./artifacts \
  --device cuda \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 4 \
  --amp
```

Базовые гиперпараметры по статье для ML-20M: `emb_dim=256`, `n_blocks=4`, `n_heads=8`, `dropout=0.2`, `max_len=200`, `n_neg=256`, `lr=1e-3`, `batch=128`, `max_epochs=100`.

## Запуск на Vast.ai (RTX 5090)

Автоматический скрипт для поиска **самого дешёвого** `1x RTX_4090` (по умолчанию) с ограничениями:
- `inet_down > 300` Mb/s
- `inet_down_cost < 0.0005` $/GB (то есть дешевле $0.5 за 1 TB)
- образ `pytorch/pytorch`

```bash
python scripts/vast_run.py --api-key <VAST_API_KEY>
```

По умолчанию скрипт:
1) находит оффер,
2) поднимает инстанс,
3) копирует текущий репозиторий,
4) запускает 1 эпоху обучения на `cuda`,
5) уничтожает инстанс после завершения.

Флаг `--keep` оставляет инстанс запущенным.

Если в окружении прокси ломает TLS-цепочку (ошибка `CERTIFICATE_VERIFY_FAILED`), используйте:

```bash
python scripts/vast_run.py --api-key <VAST_API_KEY> --insecure-ssl
```

`--insecure-ssl` отключает проверку TLS **только для вызовов Vast API** внутри скрипта (через локальный wrapper).
Для диагностики без запуска инстанса:

```bash
python scripts/vast_run.py --api-key <VAST_API_KEY> --insecure-ssl --search-only
```


Для смены модели GPU используйте флаг, например:

```bash
python scripts/vast_run.py --api-key <VAST_API_KEY> --gpu-name RTX_5090
```


## Apples-to-apples (S3Rec-style eval protocol)

Для честного сравнения с common academic setup (leave-one-out + 100 sampled negatives) запусти:

```bash
python -m esasrec.train \
  --ratings-path /path/to/ml-20m/ratings.csv \
  --workdir ./artifacts_s3rec_eval \
  --device cuda \
  --epochs 5 \
  --batch-size 128 \
  --amp \
  --eval-protocol s3rec \
  --eval-sampled-negatives 100
```

В этом режиме в логах будут `hr@1, hr@5, hr@10, ndcg@10, mrr`.
