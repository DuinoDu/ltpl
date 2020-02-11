all:
	swig -python -c++ pafprocess.i
	python setup.py build_ext --inplace
	rm *_wrap.*

clean:
	rm -rf build *.so pafprocess.py __pycache__

test:
	python test_pafprocess.py
