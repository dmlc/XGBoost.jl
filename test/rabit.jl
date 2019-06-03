function test_rabit()
    rabit_init()
    @test rabit_get_rank() == 0
    @test rabit_get_world_size() == 1
    @test !rabit_is_distributed()
    @test rabit_get_version_number() == 0
    rabit_finalize()
end

test_rabit()
