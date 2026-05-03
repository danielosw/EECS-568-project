import tempfile

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


if __name__ == "__main__":
	# STEP 1: load data
	base_dataset = MIMIC3Dataset(
		root="data",
		tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
		cache_dir=tempfile.TemporaryDirectory().name,
		dev=True,
	)

	epochs = 10
	patiance = 5
	learning_rate = 5e-4
	base_dataset.stats()
	# set the seed
	pyhealth.utils.set_seed(42)  # pyright: ignore[reportUnknownMemberType]
	# STEP 2: set task
	task = MortalityPredictionMIMIC3()
	sample_dataset = base_dataset.set_task(task)

	train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.6, 0.2, 0.2])
	train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
	val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
	test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

	# STEP 3: define model
	model = RNN(
		dataset=sample_dataset,
	)
	model2 = RETAIN(
		dataset=sample_dataset,
	)
	# ablation study - RETAIN without beta attention
	model3 = RETAINNoBeta(
		dataset=sample_dataset,
	)
	trainer = Trainer(model=model)
	trainer2 = Trainer(model=model2)
	trainer3 = Trainer(model=model3)
	trainer.train(
		train_dataloader=train_dataloader,
		val_dataloader=val_dataloader,
		epochs=epochs,
		patience=patiance,
		monitor="roc_auc",
	)
	trainer2.train(
		train_dataloader=train_dataloader,
		val_dataloader=val_dataloader,
		epochs=epochs,
		patience=patiance,
		monitor="roc_auc",
		optimizer_params={"lr": learning_rate},
	)
	trainer3.train(
		train_dataloader=train_dataloader,
		val_dataloader=val_dataloader,
		epochs=epochs,
		patience=patiance,
		monitor="roc_auc",
		optimizer_params={"lr": learning_rate},
	)

	# STEP 5: evaluate
	print("RNN Results:")
	print(trainer.evaluate(test_dataloader))
	print("RETAIN Results:")
	print(trainer2.evaluate(test_dataloader))
	print("RETAIN (No Beta) Results:")
	print(trainer3.evaluate(test_dataloader))
