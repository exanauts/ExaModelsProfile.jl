using ExaModelsExamples, CUDA, MadNLP, MadNLPGPU
using Profile
using PProf

Profile.init(; n=50_000_000, delay=0.01)
Profile.clear()
m = multi_period_ac_opf_model("pglib_opf_case78484_epigrids.m"; N=6, backend=CUDABackend())
Profile.@profile madnlp(m; tol=1e-8)
Profile.pprof()
