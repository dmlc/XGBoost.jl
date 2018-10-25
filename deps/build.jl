using BinDeps
using Libdl

@BinDeps.setup

xgboost = library_dependency("xgboost", aliases = ["libxgboost.$(Libdl.dlext)"])

if haskey(ENV, "XGBOOST_BUILD_VERSION") && ENV["XGBOOST_BUILD_VERSION"] == "master"
    libcheckout = `git checkout master`
    onload = "global const build_version = \"master\""
    @info "Using the latest master version of the XGBoost library"
else
    libcheckout = `git checkout v0.80`
    onload = "global const build_version = \"0.80\""
    @info "Using the latest stable version (0.80) of the XGBoost library"
end

provides(BuildProcess,
    FileRule(joinpath(BinDeps.libdir(xgboost), "libxgboost.$(Libdl.dlext)"),
        @build_steps begin
            CreateDirectory(BinDeps.srcdir(xgboost))
            @build_steps begin
                ChangeDirectory(BinDeps.srcdir(xgboost))
                `rm -rf xgboost`
                `git clone https://github.com/dmlc/xgboost.git --recursive`
            end

            @build_steps begin
                ChangeDirectory(joinpath(BinDeps.srcdir(xgboost), "xgboost"))
                libcheckout
                `bash build.sh`
                CreateDirectory(BinDeps.libdir(xgboost))
                `cp lib/libxgboost.$(Libdl.dlext) $(BinDeps.libdir(xgboost))`
            end
        end), xgboost, os = :Unix, onload = onload)

@BinDeps.install Dict(:xgboost => :_jl_libxgboost)
