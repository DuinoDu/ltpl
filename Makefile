
build:
	python setup.py build

upload:
	python setup.py bdist_wheel upload -r hobot-local

clean:
	@rm -rf build dist src/*.egg-info

test:
	python /usr/bin/nosetests -s tests --nologcapture

pep8:
	autopep8 src/ltpl --recursive -i

lint:
	pylint src/ltpl --reports=n

lintfull:
	pylint src/ltpl

install:
	python setup.py install

uninstall:
	python setup.py install --record install.log
	cat install.log | xargs rm -rf 
	@rm install.log
