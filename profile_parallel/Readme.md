# Profiling and multiprocessing

Lia's scattering code:

http://github.com/parejkoj/dust

Lia's simple demo of how to run the code, with some profiling scripts:

http://github.com/parejkoj/AHW2015

## building with cython

See the setup.py in AHW2015: that's what tells python how to build the cython-created
C- library.

```
cython -a scattering_utils.pyx
python setup.py build_ext --inplace
```
