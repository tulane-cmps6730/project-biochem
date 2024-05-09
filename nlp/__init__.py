# -*- coding: utf-8 -*-

"""Top-level package for nlp."""

__author__ = """Maddie Bonanno"""
__email__ = 'mbonanno@tulane.edu'
__version__ = '0.1.0'

# -*- coding: utf-8 -*-
import configparser
import os

# ~/.nlp/nlp.cfg will contain configuration information for the project,
# such as where data will be downloaded from.
# here is an example.
def write_default_config(path):
	# w = open(path, 'wt')
	# w.write('[data]\n')
	# w.write('url_small = https://tulane.app.box.com/file/1525744262136?s=imv58hbrpex4rjip3qsex2kvokg7dzm7\n')
    # w.write('url_big = https://tulane.box.com/s/ogw80o4d73253uqbtwk0na1oihvc2pe8\n')
    # w.write('file_small = %s%s%s\n' % (nlp_path, os.path.sep, 'rate-my-prof-data-cleaned.xlsx'))
    # w.write('file_big = %s%s%s\n' % (nlp_path, os.path.sep, 'bigeerdata-cleaned.xlsx'))
    # w.close()
    return

# Find NLP_HOME path
if 'NLP_HOME' in os.environ:
    nlp_path = os.environ['NLP_HOME']
else:
    nlp_path = os.environ['HOME'] + os.path.sep + '.nlp' + os.path.sep

# Make nlp directory if not present
try:
    os.makedirs(nlp_path)
except:
    pass

# main config file.
config_path = nlp_path + 'nlp.cfg'
# classifier
clf_path = nlp_path + 'clf.pkl'

# write default config if not present.
if not os.path.isfile(config_path):
    write_default_config(config_path)

# config variable now accessible throughout project.
config = configparser.RawConfigParser()
config.read(config_path)