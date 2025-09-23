try:
	from sdxl_train_network import setup_parser, SdxlNetworkTrainer
	from library.train_util import read_config_from_file

	parser = setup_parser()
	args = parser.parse_args()
	args = read_config_from_file(args, parser)

	# -------- FLOWRL INJECT (env-driven; no parser changes required) --------
	import os
	def _get_env(name, default=None): return os.getenv(name, default)
	# enable with: export FLOWRL_ENABLE=1
	setattr(args, "flowrl_enable", _get_env("FLOWRL_ENABLE", "0") == "1")
	# optional knobs
	setattr(args, "flowrl_temperature", float(_get_env("FLOWRL_TEMPERATURE", "0.9")))
	setattr(args, "flowrl_candidates", int(_get_env("FLOWRL_CANDIDATES", "6")))
	setattr(args, "flowrl_window", int(_get_env("FLOWRL_WINDOW", "1024")))
	setattr(args, "flowrl_diversity_bonus", float(_get_env("FLOWRL_DIVERSITY", "0.03")))
	setattr(args, "flowrl_tag_regex", _get_env("FLOWRL_TAG_REGEX", None))
	# ------------------------------------------------------------------------

	trainer = SdxlNetworkTrainer()
	trainer.train(args)

except BaseException:
	import sys
	import traceback
	import re
	from pygments import formatters, highlight, lexers
	from dracula import DraculaStyle

	tb = traceback.format_exc().split("\n")
	error_index = len(tb)
	for i, line in enumerate(tb):
		if re.match(r"^[A-Za-z-_]+Error:", line):
			error_index = i
			break
	tb_text = "\n".join(tb[:error_index])

	lexer = lexers.get_lexer_by_name("pytb", stripall=True)
	formatter = formatters.Terminal256Formatter(style=DraculaStyle)
	tb_colored = highlight(tb_text, lexer, formatter)

	print(f"\n{tb_colored}")
	if error_index < len(tb):
		tb_error = "\n".join(tb[error_index:])
		print(f"\033[0;31m\033[1m{tb_error}\n")

	sys.exit(1)
