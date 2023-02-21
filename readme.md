# Architecture
The proposed architecture is based on ColBERT where the document encoder is replaced by a GNN.
<center><img src="architecture.svg" width="40%"></center>


# Examples:
Let `NAME` be the name of the run and `[path-to-dataset]` be the path to the dataset in h5 format. The distillation may be trained on for 20 epochs on 2 GPUs via:

## Training
```
./distillation.py \
	run_name=NAME \
	used_gpus=[0,1,2,3] \
	trainer.devices=2 \
	trainer.accelerator=gpu \
	datamodule.data_dir="[path-to-dataset]" \
	datamodule.fold_name="fold_0" \
	datamodule.batch_size=10 \
	trainer.limit_val_batches=0 \
	trainer.max_epochs=20 \
	trainer.strategy.find_unused_parameters=True
```
Also note `used_gpus=[0,1,2,3]` which sets the `CUDA_VISIBLE_DEVICES` environment variable accordingly.

## Evaluation
```
./trec_eval.py \
	run_name=NAME \
	checkpoint=CHECKPOINT \
	trainer.accelerator=gpu \
	trainer.precision=32 \
	datamodule=trec19pass \
	datamodule.data_dir="[path-to-dataset]" \
	datamodule.num_workers=12 \
	topk=0.6
```

## Hyperparameter optimization
```
./hpopt.py \
	run_name=NAME \
	trainer.devices=1 \
	trainer.accelerator=gpu \
	trainer.precision=16 \
	trainer.max_epochs=2 \
	datamodule=trec19pass \
	datamodule.data_dir="[path-to-dataset]"
```