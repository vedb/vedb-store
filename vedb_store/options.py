import os
import appdirs
import configparser

cwd = os.path.dirname(__file__)
userdir = appdirs.user_config_dir("vedb_store", appauthor="VEDB team")
usercfg = os.path.join(userdir, "options.cfg")

config = configparser.ConfigParser()
config.read(os.path.join(cwd, 'defaults.cfg'))

# Update defaults with user-sepecifed values in user config
files_successfully_read = config.read(usercfg)

# If user config doesn't exist, create it
if len(files_successfully_read) == 0:
	os.makedirs(userdir)
	with open(usercfg, 'w') as fp:
		config.write(fp)