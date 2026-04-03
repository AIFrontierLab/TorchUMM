def run_unigame_train(cfg, config_path=None):
	# Lazy import avoids pulling heavy training dependencies when importing
	# utility modules under this package (e.g. standalone eval scripts).
	from umm.post_training.unigame.pipeline import run_unigame_train as _impl

	return _impl(cfg, config_path=config_path)


__all__ = ["run_unigame_train"]
