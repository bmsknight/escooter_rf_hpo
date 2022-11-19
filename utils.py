from sklearn import metrics


class Evaluation:
    def __init__(self, actuals, predictions):
        self.rmse = metrics.mean_squared_error(actuals, predictions, squared=False)
        self.mse = metrics.mean_squared_error(actuals, predictions)
        self.mae = metrics.mean_absolute_error(actuals, predictions)
        self.mape = metrics.mean_absolute_percentage_error(actuals, predictions)

    def print(self):
        print("MSE\tRMSE\tMAE\tMAPE")
        print("%.4f\t%.4f\t%.4f\t%.4f" % (self.mse, self.rmse, self.mae, self.mape))
