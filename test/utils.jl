
function testfilepath(name::AbstractString) 
    dir = joinpath(dirname(pathof(XGBoost)), "..")
    joinpath(dir,"assets","data",name)
end

function readlibsvm(fname::AbstractString, shape)
    dmx = zeros(Float32, shape)
    label = Float32[]
    open(fname, "r") do fi
        cnt = 1
        for line ∈ eachline(fi)
            line = split(line, " ")
            push!(label, parse(Float64, line[1]))
            line = line[2:end]
            for itm in line
                itm = split(itm, ":")
                dmx[cnt, parse(Int, itm[1]) + 1] = float(parse(Int, itm[2]))
            end
            cnt += 1
        end
    end
    (dmx, label)
end
