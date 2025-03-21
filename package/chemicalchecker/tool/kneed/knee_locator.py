import numpy as np
from scipy import interpolate
from scipy.signal import argrelextrema
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings
from typing import Tuple, Optional, Iterable
from numpy import matlib

class SimpleKneeLocator(object):
    def __init__(self, x, y, curve="concave", direction="increasing"):
        if curve != "concave" and curve != "convex":
            raise Exception("curve must be concave or convex")
        if direction != "increasing" and direction != "decreasing":
            raise Exception("direction must be increasing or decreasing")
        self.curve = curve
        self.direction = direction
        self.x = np.array(x)
        self.y = np.array(y)
        curve = np.array(y)
        idxs = np.array([i for i in range(0, len(curve))])
        if self.direction == "increasing" and self.curve == "concave":
            pass
        if self.direction == "decreasing" and self.curve == "concave":
            curve = curve[::-1]
            idxs = idxs[::-1]
        if self.direction == "decreasing" and self.curve == "convex":
            curve = -curve
        if self.direction == "increasing" and self.curve == "convex":
            curve = -curve
            curve = curve[::-1]
            idxs = idxs[::-1]
        nPoints = len(curve)
        allCoord = np.vstack((range(nPoints), curve)).T
        firstPoint = allCoord[0]
        lineVec = allCoord[-1] - allCoord[0]
        lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
        vecFromFirst = allCoord - firstPoint
        scalarProduct = np.sum(
            vecFromFirst * matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
        vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
        vecToLine = vecFromFirst - vecFromFirstParallel
        distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
        idxOfBestPoint = np.argmax(distToLine)
        self.elbow_idx = idxs[idxOfBestPoint]
        
class KneeLocator(object):
    def __init__(
        self,
        x: Iterable[float],
        y: Iterable[float],
        S: float = 1.0,
        curve: str = "concave",
        direction: str = "increasing",
        interp_method: str = "interp1d",
        online: bool = False,
    ):
        """
        Once instantiated, this class attempts to find the point of maximum
        curvature on a line. The knee is accessible via the `.knee` attribute.
        :param x: x values.
        :param y: y values.
        :param S: Sensitivity, original paper suggests default of 1.0
        :param curve: If 'concave', algorithm will detect knees. If 'convex', it
            will detect elbows.
        :param direction: one of {"increasing", "decreasing"}
        :param interp_method: one of {"interp1d", "polynomial"}
        :param online: Will correct old knee points if True, will return first knee if False
        """
        # Step 0: Raw Input
        self.x = np.array(x)
        self.y = np.array(y)
        self.curve = curve
        self.direction = direction
        self.N = len(self.x)
        self.S = S
        self.all_knees = set()
        self.all_norm_knees = set()
        self.all_knees_y = []
        self.all_norm_knees_y = []
        self.online = online

        # Step 1: fit a smooth line
        if interp_method == "interp1d":
            uspline = interpolate.interp1d(self.x, self.y)
            self.Ds_y = uspline(self.x)
        elif interp_method == "polynomial":
            pn_model = PolynomialFeatures(7)
            xpn = pn_model.fit_transform(self.x.reshape(-1, 1))
            regr_model = LinearRegression()
            regr_model.fit(xpn, self.y)
            self.Ds_y = regr_model.predict(
                pn_model.fit_transform(self.x.reshape(-1, 1))
            )
        else:
            warnings.warn(
                "{} is an invalid interp_method parameter, use either 'interp1d' or 'polynomial'".format(
                    interp_method
                )
            )
            return

        # Step 2: normalize values
        self.x_normalized = self.__normalize(self.x)
        self.y_normalized = self.__normalize(self.Ds_y)

        # Step 3: Calculate the Difference curve
        self.x_normalized, self.y_normalized = self.transform_xy(
            self.x_normalized, self.y_normalized, self.direction, self.curve
        )
        # normalized difference curve
        self.y_difference = self.y_normalized - self.x_normalized
        self.x_difference = self.x_normalized.copy()

        # Step 4: Identify local maxima/minima
        # local maxima
        self.maxima_indices = argrelextrema(self.y_difference, np.greater_equal)[0]
        self.x_difference_maxima = self.x_difference[self.maxima_indices]
        self.y_difference_maxima = self.y_difference[self.maxima_indices]

        # local minima
        self.minima_indices = argrelextrema(self.y_difference, np.less_equal)[0]
        self.x_difference_minima = self.x_difference[self.minima_indices]
        self.y_difference_minima = self.y_difference[self.minima_indices]

        # Step 5: Calculate thresholds
        self.Tmx = self.y_difference_maxima - (
            self.S * np.abs(np.diff(self.x_normalized).mean())
        )

        # Step 6: find knee
        self.knee, self.norm_knee = self.find_knee()
        self.knee_y = self.y[self.x == self.knee][0]
        self.norm_knee_y = self.y_normalized[self.x_normalized == self.norm_knee][0]

    @staticmethod
    def __normalize(a: Iterable[float]) -> Iterable[float]:
        """normalize an array
        :param a: The array to normalize
        """
        return (a - min(a)) / (max(a) - min(a))

    @staticmethod
    def transform_xy(
        x: Iterable[float], y: Iterable[float], direction: str, curve: str
    ) -> Tuple[Iterable[float], Iterable[float]]:
        """transform x and y to concave, increasing based on given direction and curve"""
        # convert elbows to knees
        if curve == "convex":
            x = x.max() - x
            y = y.max() - y
        # flip decreasing functions to increasing
        if direction == "decreasing":
            y = np.flip(y, axis=0)

        if curve == "convex":
            x = np.flip(x, axis=0)
            y = np.flip(y, axis=0)

        return x, y

    def find_knee(self,):
        """This function finds and sets the knee value and the normalized knee value. """
        if not self.maxima_indices.size:
            warnings.warn(
                "No local maxima found in the difference curve\n"
                "The line is probably not polynomial, try plotting\n"
                "the difference curve with plt.plot(knee.x_difference, knee.y_difference)\n"
                "Also check that you aren't mistakenly setting the curve argument",
                RuntimeWarning,
            )
            return None, None

        # placeholder for which threshold region i is located in.
        maxima_threshold_index = 0
        minima_threshold_index = 0
        # traverse the difference curve
        for i, x in enumerate(self.x_difference):
            # skip points on the curve before the the first local maxima
            if i < self.maxima_indices[0]:
                continue

            j = i + 1

            # reached the end of the curve
            if x == 1.0:
                break

            # if we're at a local max, increment the maxima threshold index and continue
            if (self.maxima_indices == i).any():
                threshold = self.Tmx[maxima_threshold_index]
                threshold_index = i
                maxima_threshold_index += 1
            # values in difference curve are at or after a local minimum
            if (self.minima_indices == i).any():
                threshold = 0.0
                minima_threshold_index += 1

            if self.y_difference[j] < threshold:
                if self.curve == "convex":
                    if self.direction == "decreasing":
                        knee = self.x[threshold_index]
                        norm_knee = self.x_normalized[threshold_index]
                    else:
                        knee = self.x[-(threshold_index + 1)]
                        norm_knee = self.x_normalized[-(threshold_index + 1)]

                elif self.curve == "concave":
                    if self.direction == "decreasing":
                        knee = self.x[-(threshold_index + 1)]
                        norm_knee = self.x_normalized[-(threshold_index + 1)]
                    else:
                        knee = self.x[threshold_index]
                        norm_knee = self.x_normalized[threshold_index]

                self.all_knees.add(knee)
                self.all_norm_knees.add(norm_knee)

                self.all_knees_y.append(self.y[self.x == knee][0])
                self.all_norm_knees_y.append(
                    self.y_normalized[self.x_normalized == norm_knee][0]
                )

                # if detecting in offline mode, return the first knee found
                if self.online is False:
                    return knee, norm_knee

        if self.all_knees == set():
            warnings.warn("No knee/elbow found")
            return None, None

        return knee, norm_knee

    def plot_knee_normalized(self, figsize: Optional[Tuple[int, int]] = None):
        """Plot the normalized curve, the difference curve (x_difference, y_normalized) and the knee, if it exists.

        :param figsize: Optional[Tuple[int, int]
        The figure size of the plot. Example (12, 8)
        :return: NoReturn
        """
        import matplotlib.pyplot as plt

        if figsize is None:
            figsize = (6, 6)

        plt.figure(figsize=figsize)
        plt.title("Normalized Knee Point")
        plt.plot(self.x_normalized, self.y_normalized, "b", label="normalized curve")
        plt.plot(self.x_difference, self.y_difference, "r", label="difference curve")
        plt.xticks(
            np.arange(self.x_normalized.min(), self.x_normalized.max() + 0.1, 0.1)
        )
        plt.yticks(
            np.arange(self.y_difference.min(), self.y_normalized.max() + 0.1, 0.1)
        )

        plt.vlines(
            self.norm_knee,
            plt.ylim()[0],
            plt.ylim()[1],
            linestyles="--",
            label="knee/elbow",
        )
        plt.legend(loc="best")

    def plot_knee(self, figsize: Optional[Tuple[int, int]] = None):
        """
        Plot the curve and the knee, if it exists

        :param figsize: Optional[Tuple[int, int]
            The figure size of the plot. Example (12, 8)
        :return: NoReturn
        """
        import matplotlib.pyplot as plt

        if figsize is None:
            figsize = (6, 6)

        plt.figure(figsize=figsize)
        plt.title("Knee Point")
        plt.plot(self.x, self.y, "b", label="data")
        plt.vlines(
            self.knee, plt.ylim()[0], plt.ylim()[1], linestyles="--", label="knee/elbow"
        )
        plt.legend(loc="best")

    # Niceties for users working with elbows rather than knees
    @property
    def elbow(self):
        return self.knee

    @property
    def norm_elbow(self):
        return self.norm_knee

    @property
    def elbow_y(self):
        return self.knee_y

    @property
    def norm_elbow_y(self):
        return self.norm_knee_y

    @property
    def all_elbows(self):
        return self.all_knees

    @property
    def all_norm_elbows(self):
        return self.all_norm_knees

    @property
    def all_elbows_y(self):
        return self.all_knees_y

    @property
    def all_norm_elbows_y(self):
        return self.all_norm_knees_y
