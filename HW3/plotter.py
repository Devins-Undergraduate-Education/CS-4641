###############################
### DO NOT CHANGE THIS FILE ###
###############################

# Helper functions to visualize sample regression data

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import pyplot as plt

N_SAMPLES = 700
PERCENT_TRAIN = 0.8


class Plotter:
    def __init__(
        self,
        regularization,
        poly_degree,
        print_images=False,
    ):
        self.reg = regularization
        self.POLY_DEGREE = poly_degree
        self.print_images = print_images

        self.rng = np.random.RandomState(seed=10)

        # Render types : 'browser', 'png', 'plotly_mimetype', 'jupyterlab', pdf
        # rndr_type = "jupyterlab+png"
        # pio.renderers.default = rndr_type

    def init_figure(self, title):
        # Intialize a base figure with desired formatting
        figure = go.Figure()

        camera = dict(eye=dict(x=1, y=-1.90, z=0.8), up=dict(x=0, y=0, z=1))

        figure.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Feature 1",
                yaxis_title="Feature 2",
                zaxis_title="Y",
                camera=camera,
            ),
            scene_aspectmode="cube",
            height=700,
            width=800,
            autosize=True,
        )

        return figure

    def print_figure(self, figure, title):
        fig_title = title.replace(" ", "_")
        path = f"outputs/{fig_title}.png"
        figure.write_image(path)
        img = mpimg.imread(path)
        plt.imshow(img)
        plt.axis("off")  # Turn off axis numbers and ticks
        plt.show()

    def create_data(self):
        rng = self.rng

        # Simulating a regression dataset with polynomial features.
        true_weight = rng.rand(self.POLY_DEGREE**2 + 2, 1)
        x_feature1 = np.linspace(-5, 5, N_SAMPLES)
        x_feature2 = np.linspace(-3, 3, N_SAMPLES)
        x_all = np.stack((x_feature1, x_feature2), axis=1)

        reg = self.reg
        x_all_feat = reg.construct_polynomial_feats(x_all, self.POLY_DEGREE)
        x_cart_flat = []
        for i in range(x_all_feat.shape[0]):
            point = x_all_feat[i]
            x1 = point[:, 0]
            x2 = point[:, 1]
            x1_end = x1[-1]
            x2_end = x2[-1]
            x1 = x1[:-1]
            x2 = x2[:-1]
            x3 = np.asarray([[m * n for m in x1] for n in x2])

            x3_flat = list(np.reshape(x3, (x3.shape[0] ** 2)))
            x3_flat.append(x1_end)
            x3_flat.append(x2_end)
            x3_flat = np.asarray(x3_flat)
            x_cart_flat.append(x3_flat)

        x_cart_flat = np.asarray(x_cart_flat)
        x_cart_flat = (x_cart_flat - np.mean(x_cart_flat)) / np.std(
            x_cart_flat
        )  # Normalize
        x_all_feat = np.copy(x_cart_flat)

        p = np.reshape(np.dot(x_cart_flat, true_weight), (N_SAMPLES,))
        # We must add noise to data, else the data will look unrealistically perfect.
        y_noise = rng.randn(x_all_feat.shape[0], 1)
        y_all = np.dot(x_cart_flat, true_weight) + y_noise
        print(
            "x_all: ",
            x_all.shape[0],
            " (rows/samples) ",
            x_all.shape[1],
            " (columns/features)",
            sep="",
        )
        print(
            "y_all: ",
            y_all.shape[0],
            " (rows/samples) ",
            y_all.shape[1],
            " (columns/features)",
            sep="",
        )

        return x_all, y_all, p, x_all_feat

    def split_data(self, x_all, y_all):
        rng = self.rng

        # Generate Train/Test Split
        all_indices = rng.permutation(N_SAMPLES)  # Random indicies
        train_indices = all_indices[: round(N_SAMPLES * PERCENT_TRAIN)]  # 80% Training
        test_indices = all_indices[round(N_SAMPLES * PERCENT_TRAIN) :]  # 20% Testing

        xtrain = x_all[train_indices]
        ytrain = y_all[train_indices]
        xtest = x_all[test_indices]
        ytest = y_all[test_indices]

        return xtrain, ytrain, xtest, ytest, train_indices, test_indices

    def plot_all_data(self, x_all, y_all, p):
        # Create a Dataframe
        df = pd.DataFrame(
            {
                "feature1": x_all[:, 0],
                "feature2": x_all[:, 1],
                "y": np.squeeze(y_all),
                "best_fit": np.squeeze(p),
            }
        )

        # Initialize Figure
        title = "All Simulated Datapoints"
        fig = self.init_figure(title)

        # Add scatter points to the figure with a legend name
        fig.add_scatter3d(
            x=df["feature1"],
            y=df["feature2"],
            z=df["y"],
            mode="markers",
            marker=dict(color="blue", size=8, opacity=0.12),
            name="Data Points",
        )

        # Add the line of best fit to the figure
        fig.add_scatter3d(
            x=df["feature1"],
            y=df["feature2"],
            z=df["best_fit"],
            mode="lines",
            line=dict(color="red", width=7),
            name="Line of Best Fit",
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[df["feature1"].min(), df["feature1"].max()]),
                yaxis=dict(range=[df["feature2"].min(), df["feature2"].max()]),
                zaxis=dict(
                    range=[
                        min(df["y"].min(), df["best_fit"].min()),
                        max(df["y"].max(), df["best_fit"].max()),
                    ]
                ),
            ),
            width=750,
            height=600,  # Fixed size for consistency in export
        )

        # Show the figure
        config = {"scrollZoom": True}
        fig.show(config=config)

        if self.print_images:
            self.print_figure(title)

    def plot_split_data(self, xtrain, xtest, ytrain, ytest):
        # Create a DataFrame
        ytrain = ytrain.reshape(-1)
        ytest = ytest.reshape(-1)
        train_df = pd.DataFrame(
            {
                "feature1": xtrain[:, 0],
                "feature2": xtrain[:, 1],
                "y": ytrain,
                "label": "Training",
            }
        )
        test_df = pd.DataFrame(
            {
                "feature1": xtest[:, 0],
                "feature2": xtest[:, 1],
                "y": ytest,
                "label": "Testing",
            }
        )

        # Initialize the Plotly figure
        title = "Data Set Split"
        fig = self.init_figure(title)

        # Add training data
        fig.add_scatter3d(
            x=train_df["feature1"],
            y=train_df["feature2"],
            z=train_df["y"],
            mode="markers",
            marker=dict(color="yellow", size=2, opacity=0.75),
            name="Training",
        )

        # Add testing data
        fig.add_scatter3d(
            x=test_df["feature1"],
            y=test_df["feature2"],
            z=test_df["y"],
            mode="markers",
            marker=dict(color="red", size=2, opacity=0.75),
            name="Testing",
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    range=[
                        min(train_df["feature1"].min(), test_df["feature1"].min()),
                        max(train_df["feature1"].max(), test_df["feature1"].max()),
                    ]
                ),
                yaxis=dict(
                    range=[
                        min(train_df["feature2"].min(), test_df["feature2"].min()),
                        max(train_df["feature2"].max(), test_df["feature2"].max()),
                    ]
                ),
                zaxis=dict(
                    range=[
                        min(train_df["y"].min(), test_df["y"].min()),
                        max(train_df["y"].max(), test_df["y"].max()),
                    ]
                ),
            ),
            width=750,
            height=600,
        )

        # Show the figure
        fig.show()

        # Create and print static images
        if self.print_images:
            self.print_figure(title)

    def plot_linear_closed(self, xtrain, xtest, ytrain, ytest, x_all, y_pred):
        # Create a DataFrame
        ytrain = ytrain.reshape(-1)
        ytest = ytest.reshape(-1)
        train_df = pd.DataFrame(
            {
                "feature1": xtrain[:, 0],
                "feature2": xtrain[:, 1],
                "y": ytrain,
                "label": "Training",
            }
        )
        test_df = pd.DataFrame(
            {
                "feature1": xtest[:, 0],
                "feature2": xtest[:, 1],
                "y": ytest,
                "label": "Testing",
            }
        )
        pred_df = pd.DataFrame(
            {
                "feature1": x_all[:, 0],
                "feature2": x_all[:, 1],
                "Trendline": np.squeeze(y_pred),
            }
        )

        # Initialize the Plotly figure
        title = "Linear (Closed)"
        fig = self.init_figure(title)

        # Add training data
        fig.add_scatter3d(
            x=train_df["feature1"],
            y=train_df["feature2"],
            z=train_df["y"],
            mode="markers",
            marker=dict(color="yellow", size=2, opacity=0.75),
            name="Training",
        )
        # Add testing data
        fig.add_scatter3d(
            x=test_df["feature1"],
            y=test_df["feature2"],
            z=test_df["y"],
            mode="markers",
            marker=dict(color="red", size=2, opacity=0.75),
            name="Testing",
        )
        # Add fitting line
        fig.add_scatter3d(
            x=pred_df["feature1"],
            y=pred_df["feature2"],
            z=pred_df["Trendline"],
            mode="lines",
            line=dict(color="red", width=7),
            name="Trendline",
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[x_all[:, 0].min(), x_all[:, 0].max()]),
                yaxis=dict(range=[x_all[:, 1].min(), x_all[:, 1].max()]),
                zaxis=dict(
                    range=[
                        min(ytrain.min(), ytest.min(), pred_df["Trendline"].min()),
                        max(ytrain.max(), ytest.max(), pred_df["Trendline"].max()),
                    ]
                ),
            ),
            width=750,
            height=600,
        )

        # Show the figure
        fig.show()

        # Create and print static images
        if self.print_images:
            self.print_figure(title)

    def plot_linear_gd(self, xtrain, xtest, ytrain, ytest, x_all, y_pred):
        # Create a DataFrame
        ytrain = ytrain.reshape(-1)
        ytest = ytest.reshape(-1)
        train_df = pd.DataFrame(
            {
                "feature1": xtrain[:, 0],
                "feature2": xtrain[:, 1],
                "y": ytrain,
                "label": "Training",
            }
        )
        test_df = pd.DataFrame(
            {
                "feature1": xtest[:, 0],
                "feature2": xtest[:, 1],
                "y": ytest,
                "label": "Testing",
            }
        )
        pred_df = pd.DataFrame(
            {
                "feature1": x_all[:, 0],
                "feature2": x_all[:, 1],
                "Trendline": np.squeeze(y_pred),
            }
        )

        # Initialize the Plotly figure
        title = "Linear (GD)"
        fig = self.init_figure(title)

        # Add training data
        fig.add_scatter3d(
            x=train_df["feature1"],
            y=train_df["feature2"],
            z=train_df["y"],
            mode="markers",
            marker=dict(color="yellow", size=2, opacity=0.75),
            name="Training",
        )
        # Add testing data
        fig.add_scatter3d(
            x=test_df["feature1"],
            y=test_df["feature2"],
            z=test_df["y"],
            mode="markers",
            marker=dict(color="red", size=2, opacity=0.75),
            name="Testing",
        )
        # Add fitting line
        fig.add_scatter3d(
            x=pred_df["feature1"],
            y=pred_df["feature2"],
            z=pred_df["Trendline"],
            mode="lines",
            line=dict(color="red", width=7),
            name="Trendline",
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[x_all[:, 0].min(), x_all[:, 0].max()]),
                yaxis=dict(range=[x_all[:, 1].min(), x_all[:, 1].max()]),
                zaxis=dict(
                    range=[
                        min(ytrain.min(), ytest.min(), pred_df["Trendline"].min()),
                        max(ytrain.max(), ytest.max(), pred_df["Trendline"].max()),
                    ]
                ),
            ),
            width=750,
            height=600,
        )

        # Show the figure
        fig.show()

        if self.print_images:
            self.print_figure(title)

    def plot_linear_gd_tuninglr(
        self, xtrain, xtest, ytrain, ytest, x_all, x_all_feat, learning_rates, weights
    ):
        # Create a DataFrame
        ytrain = ytrain.reshape(-1)
        ytest = ytest.reshape(-1)
        train_df = pd.DataFrame(
            {
                "feature1": xtrain[:, 0],
                "feature2": xtrain[:, 1],
                "y": ytrain,
                "label": "Training",
            }
        )
        test_df = pd.DataFrame(
            {
                "feature1": xtest[:, 0],
                "feature2": xtest[:, 1],
                "y": ytest,
                "label": "Testing",
            }
        )

        # Initialize the Plotly figure
        title = "Tuning Linear (GD)"
        fig = self.init_figure(title)

        # Add training data
        fig.add_scatter3d(
            x=train_df["feature1"],
            y=train_df["feature2"],
            z=train_df["y"],
            mode="markers",
            marker=dict(color="yellow", size=2, opacity=0.75),
            name="Training",
        )
        # Add testing data
        fig.add_scatter3d(
            x=test_df["feature1"],
            y=test_df["feature2"],
            z=test_df["y"],
            mode="markers",
            marker=dict(color="red", size=2, opacity=0.75),
            name="Testing",
        )
        # Add fitting lines
        colors = ["green", "blue", "pink"]
        for ii in range(len(learning_rates)):
            y_pred = self.reg.predict(x_all_feat, weights[ii])
            y_pred = np.reshape(y_pred, (y_pred.size,))

            pred_df = pd.DataFrame(
                {
                    "feature1": x_all[:, 0],
                    "feature2": x_all[:, 1],
                    "Trendline": np.squeeze(y_pred),
                }
            )
            fig.add_scatter3d(
                x=pred_df["feature1"],
                y=pred_df["feature2"],
                z=pred_df["Trendline"],
                mode="lines",
                line=dict(color=colors[ii], width=7),
                name="Trendline LR=" + str(learning_rates[ii]),
            )

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[x_all[:, 0].min(), x_all[:, 0].max()]),
                yaxis=dict(range=[x_all[:, 1].min(), x_all[:, 1].max()]),
                zaxis=dict(
                    range=[
                        min(ytrain.min(), ytest.min(), pred_df["Trendline"].min()),
                        max(ytrain.max(), ytest.max(), pred_df["Trendline"].max()),
                    ]
                ),
            ),
            width=750,
            height=600,
        )

        # Show the figure
        fig.show()
        if self.print_images:
            self.print_figure(title)

    def plot_10_samples(self, x_all, y_all_noisy, sub_train, y_pred, title):
        # Create a DataFrame
        samples_df = pd.DataFrame(
            {
                "feature1": x_all[sub_train, 0],
                "feature2": x_all[sub_train, 1],
                "y": np.squeeze(y_all_noisy[sub_train]),
                "label": "Samples",
            }
        )
        pred_df = pd.DataFrame(
            {
                "feature1": x_all[:, 0],
                "feature2": x_all[:, 1],
                "Trendline": np.squeeze(y_pred),
            }
        )

        # Initialize the Plotly figure
        fig = self.init_figure(title)

        # Add training data
        fig.add_scatter3d(
            x=samples_df["feature1"],
            y=samples_df["feature2"],
            z=samples_df["y"],
            mode="markers",
            marker=dict(
                color="red",
                opacity=0.75,
                size=6,
                symbol="x",
                line=dict(width=1, color="red"),
            ),
            name="Samples",
        )
        # Add fitting line
        fig.add_scatter3d(
            x=pred_df["feature1"],
            y=pred_df["feature2"],
            z=pred_df["Trendline"],
            mode="lines",
            line=dict(color="blue", width=7),
            name="Trendline",
        )

        z_min = max(min(samples_df["y"].min(), pred_df["Trendline"].min()), -1000)
        z_max = max(samples_df["y"].max(), pred_df["Trendline"].max())
        print(f"z_min = {z_min}, z_max = {z_max}")

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    range=[samples_df["feature1"].min(), samples_df["feature1"].max()]
                ),
                yaxis=dict(
                    range=[samples_df["feature2"].min(), samples_df["feature2"].max()]
                ),
                zaxis=dict(range=[z_min, z_max]),
            ),
            width=750,
            height=600,
        )

        # Show the figure
        fig.show()

        if self.print_images:
            self.print_figure(title)
