"""
Outlier detection algorithms.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

# Try to import HDBSCAN
try:
    from sklearn.cluster import HDBSCAN as SK_HDBSCAN

    HAVE_SK_HDBSCAN = True
except ImportError:
    try:
        import hdbscan as HDBSCAN_PKG

        HAVE_SK_HDBSCAN = False
    except ImportError:
        raise ImportError(
            "HDBSCAN not available. Install sklearn>=1.3 or pip install hdbscan"
        )

# Try to import DBCV
try:
    from dbcv import dbcv

    HAVE_DBCV = True
except ImportError:
    HAVE_DBCV = False
    logging.warning("DBCV not available. Install with: pip install dbcv")

logger = logging.getLogger(__name__)


class OutlierDetector(ABC):
    """
    Abstract base class for outlier detectors.

    All outlier detectors must implement fit_predict method.
    """

    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit detector and predict outliers.

        Args:
            X: Feature matrix

        Returns:
            Binary array: 1 = outlier, 0 = inlier
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get detector name."""
        pass


class DBSCANDetector(OutlierDetector):
    """
    DBSCAN-based outlier detector.

    Automatically determines eps using k-distance plot.
    """

    def __init__(
        self,
        k: int = 5,
        eps_percentile: int = 90,
        min_samples: int = 5,
        show_plot: bool = False,
    ):
        self.k = k
        self.eps_percentile = eps_percentile
        self.min_samples = min_samples
        self.show_plot = show_plot
        self.eps_ = None
        self.labels_ = None

    def _calculate_eps(self, X: np.ndarray) -> float:
        """
        Calculate eps using k-distance plot.

        Args:
            X: Feature matrix

        Returns:
            Optimal eps value
        """
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.k).fit(X)
        distances, _ = nbrs.kneighbors(X)

        # Get k-th nearest neighbor distance for each point
        k_distances = distances[:, -1]
        k_distances_sorted = np.sort(k_distances)

        # Use percentile as eps
        eps_value = float(np.percentile(k_distances_sorted, self.eps_percentile))

        logger.info(f"DBSCAN eps (p{self.eps_percentile}) = {eps_value:.4f}")

        # Plot if requested
        if self.show_plot:
            plt.figure(figsize=(8, 4))
            plt.plot(k_distances_sorted)
            plt.axhline(
                eps_value, color="red", linestyle="--", label=f"eps={eps_value:.4f}"
            )
            plt.xlabel("Points (sorted by distance)")
            plt.ylabel(f"{self.k}-NN Distance")
            plt.title(f"K-Distance Plot (k={self.k})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        return eps_value

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit DBSCAN and predict outliers."""
        # Calculate eps
        self.eps_ = self._calculate_eps(X)

        # Run DBSCAN
        dbscan = DBSCAN(eps=self.eps_, min_samples=self.min_samples)
        self.labels_ = dbscan.fit_predict(X)

        # Outliers are labeled as -1
        outliers = (self.labels_ == -1).astype(int)

        n_outliers = outliers.sum()
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)

        logger.info(
            f"DBSCAN: {n_clusters} clusters, {n_outliers} outliers "
            f"({n_outliers/len(X)*100:.1f}%)"
        )

        return outliers

    def get_name(self) -> str:
        return "DBSCAN"


class HDBSCANDetector(OutlierDetector):
    """
    HDBSCAN-based outlier detector.

    Hierarchical density-based clustering, more robust than DBSCAN.
    """

    def __init__(self, min_cluster_size: int = 5, min_samples: int = 3):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.labels_ = None
        self.outlier_scores_ = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit HDBSCAN and predict outliers."""
        # Use sklearn's HDBSCAN if available, otherwise fallback
        if HAVE_SK_HDBSCAN:
            clusterer = SK_HDBSCAN(
                min_cluster_size=self.min_cluster_size, min_samples=self.min_samples
            )
        else:
            clusterer = HDBSCAN_PKG.HDBSCAN(
                min_cluster_size=self.min_cluster_size, min_samples=self.min_samples
            )

        self.labels_ = clusterer.fit_predict(X)

        # Get outlier scores if available
        if hasattr(clusterer, "outlier_scores_"):
            self.outlier_scores_ = clusterer.outlier_scores_

        # Outliers are labeled as -1
        outliers = (self.labels_ == -1).astype(int)

        n_outliers = outliers.sum()
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)

        logger.info(
            f"HDBSCAN: {n_clusters} clusters, {n_outliers} outliers "
            f"({n_outliers/len(X)*100:.1f}%)"
        )

        return outliers

    def get_name(self) -> str:
        return "HDBSCAN"


class IsolationForestDetector(OutlierDetector):
    """
    Isolation Forest outlier detector.

    Tree-based method that isolates outliers by random splits.
    """

    def __init__(self, contamination: float = 0.10, random_state: Optional[int] = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.scores_ = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit Isolation Forest and predict outliers."""
        if_model = IsolationForest(
            contamination=self.contamination, random_state=self.random_state
        )

        labels = if_model.fit_predict(X)

        # Isolation Forest returns -1 for outliers, 1 for inliers
        outliers = (labels == -1).astype(int)

        # Get anomaly scores (more negative = more outlier)
        self.scores_ = if_model.score_samples(X)

        n_outliers = outliers.sum()
        logger.info(
            f"Isolation Forest: {n_outliers} outliers "
            f"({n_outliers/len(X)*100:.1f}%)"
        )

        return outliers

    def get_name(self) -> str:
        return "IsolationForest"


class DensityEnsemble:
    """
    Ensemble that chooses between DBSCAN and HDBSCAN using DBCV.

    If DBCV is not available, defaults to HDBSCAN if it found outliers,
    otherwise DBSCAN.
    """

    def __init__(
        self,
        dbscan_detector: DBSCANDetector,
        hdbscan_detector: HDBSCANDetector,
        use_dbcv: bool = True,
    ):
        self.dbscan_detector = dbscan_detector
        self.hdbscan_detector = hdbscan_detector
        self.use_dbcv = use_dbcv
        self.chosen_method_ = None
        self.dbcv_scores_ = {}

    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Fit both detectors and choose best one.

        Returns:
            Tuple of (outliers, metadata)
        """
        # Run both detectors
        dbscan_outliers = self.dbscan_detector.fit_predict(X)
        hdbscan_outliers = self.hdbscan_detector.fit_predict(X)

        # Calculate DBCV scores if available and requested
        if self.use_dbcv and HAVE_DBCV:
            dbcv_db = self._safe_dbcv(X, self.dbscan_detector.labels_)
            dbcv_hdb = self._safe_dbcv(X, self.hdbscan_detector.labels_)

            self.dbcv_scores_ = {"dbscan": dbcv_db, "hdbscan": dbcv_hdb}

            # Choose method with higher DBCV
            if not np.isnan(dbcv_db) and not np.isnan(dbcv_hdb):
                if dbcv_hdb >= dbcv_db:
                    self.chosen_method_ = "HDBSCAN"
                    chosen_outliers = hdbscan_outliers
                else:
                    self.chosen_method_ = "DBSCAN"
                    chosen_outliers = dbscan_outliers

                logger.info(
                    f"Density method selected by DBCV: {self.chosen_method_} "
                    f"(DBSCAN={dbcv_db:.3f}, HDBSCAN={dbcv_hdb:.3f})"
                )
            elif not np.isnan(dbcv_hdb):
                self.chosen_method_ = "HDBSCAN"
                chosen_outliers = hdbscan_outliers
            else:
                self.chosen_method_ = "DBSCAN"
                chosen_outliers = dbscan_outliers
        else:
            # Fallback: use HDBSCAN if it found outliers, else DBSCAN
            if hdbscan_outliers.sum() > 0:
                self.chosen_method_ = "HDBSCAN"
                chosen_outliers = hdbscan_outliers
                logger.info("Density method: HDBSCAN (fallback)")
            else:
                self.chosen_method_ = "DBSCAN"
                chosen_outliers = dbscan_outliers
                logger.info("Density method: DBSCAN (fallback)")

        metadata = {
            "chosen_method": self.chosen_method_,
            "dbscan_outliers": dbscan_outliers,
            "hdbscan_outliers": hdbscan_outliers,
            "dbscan_labels": self.dbscan_detector.labels_,
            "hdbscan_labels": self.hdbscan_detector.labels_,
            "dbcv_scores": self.dbcv_scores_,
        }

        return chosen_outliers, metadata

    def _safe_dbcv(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate DBCV score safely."""
        unique_labels = np.unique(labels)

        # Need at least 2 clusters (excluding noise)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        if n_clusters < 2:
            return np.nan

        try:
            return float(dbcv(X, labels))
        except Exception as e:
            logger.warning(f"DBCV calculation failed: {e}")
            return np.nan
