function readlibsvm(fname::String, shape)
    dmx = zeros(Float32, shape)
    label = Float32[]
    fi = open(fname, "r")
    cnt = 1
    for line in eachline(fi)
        line = split(line, " ")
        push!(label, float(line[1]))
        line = line[2:end]
        for itm in line
            itm = split(itm, ":")
            dmx[cnt, parse(Int, itm[1]) + 1] = float(parse(Int, itm[2]))
        end
        cnt += 1
    end
    close(fi)
    return (dmx, label)
end
