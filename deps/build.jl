using BinDeps
@BinDeps.setup

deps = [ libxgboostwrapper = library_dependency("xgboostwrapper", aliases = ["libxgboost.so"]) ]

prefix=joinpath(BinDeps.depsdir(libxgboostwrapper),"usr")
provides(BuildProcess,
           (@build_steps begin
               `rm -rf xgboost`
               `git clone https://github.com/dmlc/xgboost.git --recursive`
               CreateDirectory(prefix)
               CreateDirectory(joinpath(prefix, "lib"))
               @build_steps begin
                   ChangeDirectory("xgboost")
                   FileRule(joinpath(prefix,"lib","libxgboost.so"), @build_steps begin
                       `bash build.sh`
                       `cp lib/libxgboost.so $prefix/lib`
                   end)
               end
            end),
         libxgboostwrapper)

@BinDeps.install Dict(:xgboostwrapper => :_xgboost)
