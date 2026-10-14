"""Interface shim: src/ml/* maps onto crowd_reliability.ml.* (the real package)."""
from crowd_reliability.ml.tree_bank import (  # noqa: F401
    CHANNEL_METADATA,
    TREE_CHANNELS,
    DecisionTreeBank,
)
from crowd_reliability.ml.trees import (  # noqa: F401
    TreeSnapshot,
    TreeVectorResult,
    infer_tree_vector,
)
from crowd_reliability.ml.tree_train import (  # noqa: F401
    LeakageError,
    benchmark_channels,
    build_training_set,
    monte_carlo_refit,
    train_tree_bank,
)
