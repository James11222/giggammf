```{toctree}
:maxdepth: 2
:hidden:

basic_usage
gpu_usage
api_tab

```

# GIGGA-MMF

**G**PU-based **I**dentification of structure on **G**rids of **G**i**A**nt scale - **M**ultiscale **M**orphological **F**ilter (**GIGGA-MMF**) is a python package for identifying structures in the cosmic web in massive N-body simulations. 

To install the code

```bash
pip install -e ".[gpu]"     # JAX with CUDA 12, and CuPy
```

## Authors

James Sunseri (Princeton)

### AI Acknowledgement

While this code was substantially written by LLMs, it is a direct rewrite of a predominantly human written code (`pycosmommf`). The role of the LLM was to handle the dirty work of implementing techniques to enable massive parallel scaling across many GPUs. We have tested that the code reproduces its predecessor's results to ensure reliability. The code uses common techniques to compute distributed FFTs across multiple GPUs across multiple nodes. 
