from src.duck_controller import DuckControllerConfig, prepare_task_map_data


def test_task_map_injects_separated_stop_and_dynamic_duck():
    source = {"tiles": [["straight/E"]], "tile_size": 0.585}
    cfg = DuckControllerConfig(
        inject_if_missing=True,
        inject_stop_if_missing=True,
        require_duck=True,
        require_stop=True,
    )

    prepared, duck_count, stop_count = prepare_task_map_data(source, cfg)

    assert duck_count == 1
    assert stop_count == 1
    assert "objects" not in source  # input tidak dimutasi
    duck = prepared["objects"]["mdp_duckie"]
    stop = prepared["objects"]["mdp_stop_sign"]
    assert duck["static"] is False
    assert stop["static"] is True
    assert duck["pos"] != stop["pos"]
