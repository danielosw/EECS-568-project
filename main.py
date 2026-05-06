import tempfile

from pathlib import Path

import pyhealth.utils


import torch
from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import RETAIN, RNN
from pyhealth.models.retain import RETAINLayer
from pyhealth.tasks import MortalityPredictionMIMIC3
from pyhealth.trainer import Trainer
import matplotlib.pyplot as plt

# Custom RETAINLayer without beta attention for ablation
class RETAINLayerNoBeta(RETAINLayer):
	def compute_beta(self, rx, lengths, total_length: int):
		# Return a tensor of zeros
		return torch.ones_like(rx)


# Custom RETAIN model using the no-beta layer
class RETAINNoBeta(RETAIN):
	def __init__(self, dataset, **kwargs):
		super().__init__(dataset, **kwargs)
		# Replace standard RETAINLayer with RETAINLayerNoBeta
		for feature_key in self.retain:
			self.retain[feature_key] = RETAINLayerNoBeta(
				feature_size=self.embedding_dim, dropout=0.5
			)


def train_and_collect_metrics(trainer, train_dataloader, val_dataloader, **train_kwargs):
	loss_by_epoch = []
	roc_auc_by_epoch = []
	original_evaluate = trainer.evaluate

	def evaluate_and_capture(dataloader):
		scores = original_evaluate(dataloader)

		if "loss" in scores:
			loss_by_epoch.append(scores["loss"])
		if "roc_auc" in scores:
			roc_auc_by_epoch.append(scores["roc_auc"])
		return scores

	trainer.evaluate = evaluate_and_capture
	try:
		trainer.train(
			train_dataloader=train_dataloader,
			val_dataloader=val_dataloader,
			**train_kwargs,
		)
	finally:
		trainer.evaluate = original_evaluate

	return loss_by_epoch, roc_auc_by_epoch


def save_loss_plot(loss_curves, output_path):
	plt.figure(figsize=(8, 5))
	for model_name, curve in loss_curves.items():
		epochs = list(range(1, len(curve) + 1))
		plt.plot(epochs, curve, marker="o", label=model_name)
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Validation Loss by Epoch")
	plt.legend()
	plt.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=200)
	plt.close()


def save_roc_auc_plot(roc_auc_curves, output_path):
	plt.figure(figsize=(8, 5))
	for model_name, curve in roc_auc_curves.items():
		epochs = list(range(1, len(curve) + 1))
		plt.plot(epochs, curve, marker="o", label=model_name)
	plt.xlabel("Epoch")
	plt.ylabel("ROC AUC")
	plt.title("Validation ROC AUC by Epoch")
	plt.legend()
	plt.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=200)
	plt.close()


if __name__ == "__main__":
	# STEP 1: load data
	base_dataset = MIMIC3Dataset(
		root="data",
		tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
		cache_dir=tempfile.TemporaryDirectory().name,
		dev=False,
		# set workers to half of cores
		num_workers=max(1, torch.get_num_interop_threads() // 2)

	)

	epochs = 100
	patiance = 10
	learning_rate = 5e-5
	monitor = "roc_auc"
	monitor_criterion = "max"
	ratios= [0.6, 0.2, 0.2]
	abilation = True
	# set the seed
	pyhealth.utils.set_seed(42)  # pyright: ignore[reportUnknownMemberType]
	base_dataset.stats()

	# STEP 2: set task
	task = MortalityPredictionMIMIC3()
	sample_dataset = base_dataset.set_task(task)

	train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, ratios=ratios)
	train_dataloader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
	val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
	test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

	# STEP 3: define model
	model = RNN(
		dataset=sample_dataset,
	)
	model2 = RETAIN(
		dataset=sample_dataset,
	)
	if(abilation):
		# ablation study - RETAIN without beta attention
		model3 = RETAINNoBeta(
			dataset=sample_dataset,
		)
		trainer3 = Trainer(model=model3)

	trainer = Trainer(model=model)
	trainer2 = Trainer(model=model2)
	
	# STEP 4: train
	loss_curves = {}
	roc_auc_curves = {}
	
	loss_curves["RNN"], roc_auc_curves["RNN"] = train_and_collect_metrics(
		trainer=trainer,
		train_dataloader=train_dataloader,
		val_dataloader=val_dataloader,
		epochs=epochs,
		patience=patiance,
		monitor=monitor,
		monitor_criterion=monitor_criterion,
		optimizer_params={"lr": learning_rate},

	)
	loss_curves["RETAIN"], roc_auc_curves["RETAIN"] = train_and_collect_metrics(
		trainer=trainer2,
		train_dataloader=train_dataloader,
		val_dataloader=val_dataloader,
		epochs=epochs,
		patience=patiance,
		monitor=monitor,
		monitor_criterion=monitor_criterion,
		optimizer_params={"lr": learning_rate},
	)
	if(abilation):
		loss_curves["RETAIN (No Beta)"], roc_auc_curves["RETAIN (No Beta)"] = train_and_collect_metrics(
			trainer=trainer3,
			train_dataloader=train_dataloader,
			val_dataloader=val_dataloader,
			epochs=epochs,
			patience=patiance,
			monitor=monitor,
			monitor_criterion=monitor_criterion,
			optimizer_params={"lr": learning_rate},
	)

	loss_plot_path = Path("output") / "loss_by_epoch.png"
	save_loss_plot(loss_curves, loss_plot_path)
	print(f"Saved Loss plot to: {loss_plot_path}")
	
	roc_auc_plot_path = Path("output") / "roc_auc_by_epoch.png"
	save_roc_auc_plot(roc_auc_curves, roc_auc_plot_path)
	print(f"Saved ROC AUC plot to: {roc_auc_plot_path}")

	# STEP 5: evaluate
	print("RNN Results:")
	print(trainer.evaluate(test_dataloader))
	print("RETAIN Results:")
	print(trainer2.evaluate(test_dataloader))
	if(abilation):
		print("RETAIN (No Beta) Results:")
		print(trainer3.evaluate(test_dataloader))
