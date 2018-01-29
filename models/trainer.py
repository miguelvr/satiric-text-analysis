class Trainer(object):

    def __init__(self, model, logger):
        self.logger = logger
        self.model = model

    def fit(self, train_data, dev_data, epochs=10):
        # Start trainer
        for epoch_n in range(epochs):

            # Train
            for batch in train_data:
                objective = self.model.update(**batch)
                self.logger.update_on_batch(objective)

            # Validation
            predictions = []
            gold = []
            for batch in dev_data:
                predictions.append(self.model.predict(batch['input']))
                gold.append(batch['output'])

            self.logger.update_on_epoch(predictions, gold)
            if self.logger.state == 'save':
                self.model.save()
