def run_janus_pro_train(cfg):
	# Lazy import avoids importing training stack when only eval utilities are used.
	from umm.post_training.unigame.janus_pro.train import run_janus_pro_train as _impl

	return _impl(cfg)


__all__ = ["run_janus_pro_train"]
