# Python Profiling and Simple Parallelization 

_John Parejko (@parejkoj), Lia Corrales (), Phil Marshall 
(@drphilmarshall) and Andrew Hearin (@aphearin)_

* [The Faster Python notebook](FasterPython.ipynb) contains most of the 
examples from John Parejko's breakout Multiprocessing, distilled into 
minimal working examples. It needs some more work in order for it to 
include `line_profiler` and `cython` demos, and could use some demo of 
parallelization beyond `multiprocessing`.

* For a more involved (and real world) demo, check out John's analysis 
and partial speed-up of [Lia's scattering 
code](http://github.com/parejkoj/dust) including [Lia's simple demo of 
how to run the code, with some profiling 
scripts](http://github.com/parejkoj/AHW2015)

## Building with cython

* See the `setup.py` in [John's fork of Lia's AHW2015 
repo](http://github.com/parejkoj/AHW2015): that's what tells python how 
to build the `cython`-created C- library.

```
cython -a scattering_utils.pyx
python setup.py build_ext --inplace
```

* The file `scattering_utils.html` that gets produced is what lets you 
view the C- code that `cython` produced.
