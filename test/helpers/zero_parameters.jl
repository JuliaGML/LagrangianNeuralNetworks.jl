# The parameters of `ps` set to zero, so that the network is `L ≡ 0`.
function zero_parameters!(ps)
    for layer in keys(ps), array in values(ps[layer])

        fill!(array, 0)
    end
    ps
end
