"""建筑平面图的 NetworkX 拓扑构造与可视化。

本模块不依赖 ``hazards``：只负责静态图（节点类型、几何、边长、易受害权重）。
坐标单位为米；边权 ``length`` 供最短路与灾害前锋传播使用。
"""

from __future__ import annotations

from typing import Any, Hashable, Iterable, Mapping

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes
from matplotlib.patches import Patch

NodeId = str

_OFFICE_HALL_X: tuple[float, float, float] = (10.0, 15.0, 20.0)
_OFFICE_L: float = 30.0
_OFFICE_D: float = 8.0
_STAIR_FLIGHT_M: float = 8.0
_FLOOR_RENDER_DY: float = 28.0


def _add_node(
    graph: nx.Graph,
    node_id: NodeId,
    *,
    node_type: str,
    pos: tuple[float, float],
    floor: int = 0,
    search_required: bool = False,
    vulnerable_weight: float = 0.0,
    occupancy_type: str = "none",
    extra: Mapping[str, Any] | None = None,
) -> None:
    """向图中加入带标准属性的节点。

    Parameters
    ----------
    pos :
        二维绘图/几何坐标，形状 ``(2,)``，单位 m。
    vulnerable_weight :
        易受害人群权重，标量 ``>= 0``；儿童活动区显著大于员工区。
    """
    attrs: dict[str, Any] = {
        "node_type": node_type,
        "pos": pos,
        "floor": floor,
        "search_required": search_required,
        "vulnerable_weight": float(vulnerable_weight),
        "occupancy_type": occupancy_type,
    }
    if extra:
        attrs.update(dict(extra))
    graph.add_node(node_id, **attrs)


def _add_edge(
    graph: nx.Graph,
    u: NodeId,
    v: NodeId,
    *,
    kind: str,
    length: float | None = None,
) -> None:
    """无向边；缺省 ``length`` 时用端点欧氏距离。"""
    if length is None:
        x1, y1 = graph.nodes[u]["pos"]
        x2, y2 = graph.nodes[v]["pos"]
        length = float(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)
    graph.add_edge(u, v, kind=kind, length=float(length))


def get_office_layout() -> nx.Graph:
    """复现 Problem A Figure 1：6 间办公室 + 3 走廊节点 + 东西各 1 个 Exit。

    拓扑
    ----
    ``E_W -- H1 -- H2 -- H3 -- E_E``，北室 ``N1,N2,N3`` 与南室 ``S1,S2,S3``
    分别悬挂在 ``H1,H2,H3`` 上。门位与解析基线一致：
    ``x_H1=10, x_H2=15, x_H3=20``，``E_W`` 在 ``x=0``，``E_E`` 在 ``x=L=30``。

    Returns
    -------
    nx.Graph
        无向图。节点属性含 ``node_type, pos, floor, search_required``；
        边属性含 ``kind, length``（m）。
    """
    graph = nx.Graph()
    graph.graph["layout_name"] = "office"
    graph.graph["length_unit"] = "m"
    graph.graph["description"] = "HiMCM 2025 Problem A Figure 1 office"

    _add_node(graph, "E_W", node_type="exit", pos=(0.0, 0.0), occupancy_type="exit")
    _add_node(graph, "E_E", node_type="exit", pos=(_OFFICE_L, 0.0), occupancy_type="exit")

    hall_ids = ("H1", "H2", "H3")
    for hid, x_h in zip(hall_ids, _OFFICE_HALL_X, strict=True):
        _add_node(graph, hid, node_type="hallway", pos=(x_h, 0.0), occupancy_type="circulation")

    north_ids = ("N1", "N2", "N3")
    south_ids = ("S1", "S2", "S3")
    for nid, x_h in zip(north_ids, _OFFICE_HALL_X, strict=True):
        _add_node(
            graph,
            nid,
            node_type="room",
            pos=(x_h, _OFFICE_D),
            search_required=True,
            occupancy_type="office",
        )
    for sid, x_h in zip(south_ids, _OFFICE_HALL_X, strict=True):
        _add_node(
            graph,
            sid,
            node_type="room",
            pos=(x_h, -_OFFICE_D),
            search_required=True,
            occupancy_type="office",
        )

    _add_edge(graph, "E_W", "H1", kind="hallway")
    _add_edge(graph, "H1", "H2", kind="hallway")
    _add_edge(graph, "H2", "H3", kind="hallway")
    _add_edge(graph, "H3", "E_E", kind="hallway")

    for room_id, hall_id in zip(north_ids + south_ids, hall_ids + hall_ids, strict=True):
        _add_edge(graph, room_id, hall_id, kind="room_access", length=_OFFICE_D)

    return graph


def _daycare_room_profile(slot: str) -> tuple[str, float]:
    """按开间槽位给出 occupancy_type 与 vulnerable_weight。

    权重为无量纲优先级：婴幼儿 > 午睡/学步 > 教室 > 员工。
    """
    profiles: dict[str, tuple[str, float]] = {
        "N1": ("infant", 5.0),
        "N2": ("toddler", 4.0),
        "N3": ("classroom", 3.0),
        "S1": ("classroom", 3.0),
        "S2": ("staff", 1.0),
        "S3": ("nap_room", 4.5),
    }
    return profiles[slot]


def get_daycare_layout(*, n_floors: int = 3) -> nx.Graph:
    """Req 3：三层日托平面。每层 6 室 + 3 走廊节点，楼梯竖向连通。

    Parameters
    ----------
    n_floors :
        层数，标量，默认 3。第 0 层含 ``E_W, E_E`` 对外出口。

    楼梯 ``F{f}_ST_W`` / ``F{f}_ST_E`` 挂在该层 ``H1`` / ``H3`` 上，
    相邻楼层同侧楼梯以 ``kind="stair"``、``length=_STAIR_FLIGHT_M`` 相连。
    """
    if n_floors < 1:
        raise ValueError(f"n_floors must be >= 1, got {n_floors}")

    graph = nx.Graph()
    graph.graph["layout_name"] = "daycare"
    graph.graph["length_unit"] = "m"
    graph.graph["n_floors"] = n_floors
    graph.graph["description"] = "Req 3 multi-storey daycare with child vulnerability weights"

    def y_draw(local_y: float, floor: int) -> float:
        return local_y + float(floor) * _FLOOR_RENDER_DY

    for floor in range(n_floors):
        prefix = f"F{floor}_"
        hall_ids = (f"{prefix}H1", f"{prefix}H2", f"{prefix}H3")
        for hid, x_h in zip(hall_ids, _OFFICE_HALL_X, strict=True):
            _add_node(
                graph,
                hid,
                node_type="hallway",
                pos=(x_h, y_draw(0.0, floor)),
                floor=floor,
                occupancy_type="circulation",
            )

        for slot, x_h in zip(("N1", "N2", "N3"), _OFFICE_HALL_X, strict=True):
            occ, weight = _daycare_room_profile(slot)
            _add_node(
                graph,
                f"{prefix}{slot}",
                node_type="room",
                pos=(x_h, y_draw(_OFFICE_D, floor)),
                floor=floor,
                search_required=True,
                vulnerable_weight=weight,
                occupancy_type=occ,
            )
        for slot, x_h in zip(("S1", "S2", "S3"), _OFFICE_HALL_X, strict=True):
            occ, weight = _daycare_room_profile(slot)
            _add_node(
                graph,
                f"{prefix}{slot}",
                node_type="room",
                pos=(x_h, y_draw(-_OFFICE_D, floor)),
                floor=floor,
                search_required=True,
                vulnerable_weight=weight,
                occupancy_type=occ,
            )

        _add_node(
            graph,
            f"{prefix}ST_W",
            node_type="stair",
            pos=(5.0, y_draw(0.0, floor)),
            floor=floor,
            occupancy_type="stair",
        )
        _add_node(
            graph,
            f"{prefix}ST_E",
            node_type="stair",
            pos=(25.0, y_draw(0.0, floor)),
            floor=floor,
            occupancy_type="stair",
        )

        _add_edge(graph, hall_ids[0], hall_ids[1], kind="hallway")
        _add_edge(graph, hall_ids[1], hall_ids[2], kind="hallway")
        _add_edge(graph, f"{prefix}N1", hall_ids[0], kind="room_access", length=_OFFICE_D)
        _add_edge(graph, f"{prefix}N2", hall_ids[1], kind="room_access", length=_OFFICE_D)
        _add_edge(graph, f"{prefix}N3", hall_ids[2], kind="room_access", length=_OFFICE_D)
        _add_edge(graph, f"{prefix}S1", hall_ids[0], kind="room_access", length=_OFFICE_D)
        _add_edge(graph, f"{prefix}S2", hall_ids[1], kind="room_access", length=_OFFICE_D)
        _add_edge(graph, f"{prefix}S3", hall_ids[2], kind="room_access", length=_OFFICE_D)
        _add_edge(graph, f"{prefix}ST_W", hall_ids[0], kind="hallway")
        _add_edge(graph, f"{prefix}ST_E", hall_ids[2], kind="hallway")

        if floor > 0:
            prev = f"F{floor - 1}_"
            _add_edge(
                graph,
                f"{prev}ST_W",
                f"{prefix}ST_W",
                kind="stair",
                length=_STAIR_FLIGHT_M,
            )
            _add_edge(
                graph,
                f"{prev}ST_E",
                f"{prefix}ST_E",
                kind="stair",
                length=_STAIR_FLIGHT_M,
            )

    _add_node(graph, "E_W", node_type="exit", pos=(0.0, y_draw(0.0, 0)), occupancy_type="exit")
    _add_node(
        graph,
        "E_E",
        node_type="exit",
        pos=(_OFFICE_L, y_draw(0.0, 0)),
        occupancy_type="exit",
    )
    _add_edge(graph, "E_W", "F0_ST_W", kind="hallway")
    _add_edge(graph, "E_W", "F0_H1", kind="hallway")
    _add_edge(graph, "E_E", "F0_ST_E", kind="hallway")
    _add_edge(graph, "E_E", "F0_H3", kind="hallway")

    return graph


def _warehouse_obstacles(n_rows: int, n_cols: int) -> set[tuple[int, int]]:
    """货架障碍（不可通行格），在网格中留出纵横多通道。"""
    racks: set[tuple[int, int]] = set()
    row_blocks = ((1, 2), (5, 6))
    col_blocks = ((2, 4), (7, 9))
    for r0, r1 in row_blocks:
        for c0, c1 in col_blocks:
            for row in range(r0, r1 + 1):
                for col in range(c0, c1 + 1):
                    if 0 <= row < n_rows and 0 <= col < n_cols:
                        racks.add((row, col))
    return racks


def get_warehouse_layout(
    *,
    n_rows: int = 8,
    n_cols: int = 12,
    cell_size: float = 4.0,
) -> nx.Graph:
    """Req 3：大开间仓库网格。货架为障碍，纵横通道 + 多出口。

    Parameters
    ----------
    n_rows, n_cols :
        网格行列数（标量）。默认 ``8 x 12``。
    cell_size :
        相邻四连通格点间距（m），标量，默认 4。

    四连通；贴货架的可行走格标记为 ``bay``（需搜寻），其余为 ``aisle``。
    四角及长边中点为 ``exit``，对应多通道疏散。
    """
    if n_rows < 3 or n_cols < 3:
        raise ValueError("warehouse grid must be at least 3x3")

    graph = nx.Graph()
    graph.graph["layout_name"] = "warehouse"
    graph.graph["length_unit"] = "m"
    graph.graph["cell_size"] = float(cell_size)
    graph.graph["grid_shape"] = (n_rows, n_cols)
    graph.graph["description"] = "Req 3 warehouse open bay with racks and multi-aisle exits"

    obstacles = _warehouse_obstacles(n_rows, n_cols)
    graph.graph["obstacle_cells"] = tuple(sorted(obstacles))

    def cell_id(row: int, col: int) -> NodeId:
        return f"W_{row}_{col}"

    def is_exit(row: int, col: int) -> bool:
        corners = {
            (0, 0),
            (0, n_cols - 1),
            (n_rows - 1, 0),
            (n_rows - 1, n_cols - 1),
        }
        mid_c = n_cols // 2
        mid_r = n_rows // 2
        mids = {(0, mid_c), (n_rows - 1, mid_c), (mid_r, 0), (mid_r, n_cols - 1)}
        return (row, col) in corners or (row, col) in mids

    walkable: list[tuple[int, int]] = []
    for row in range(n_rows):
        for col in range(n_cols):
            if (row, col) in obstacles:
                continue
            walkable.append((row, col))
            neighbors_4 = (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            )
            touches_rack = any(nb in obstacles for nb in neighbors_4)
            if is_exit(row, col):
                node_type = "exit"
                search = False
                occ = "exit"
            elif touches_rack:
                node_type = "bay"
                search = True
                occ = "storage"
            else:
                node_type = "aisle"
                search = False
                occ = "circulation"
            _add_node(
                graph,
                cell_id(row, col),
                node_type=node_type,
                pos=(col * cell_size, row * cell_size),
                search_required=search,
                occupancy_type=occ,
                extra={"grid_rc": (row, col)},
            )

    offsets = ((1, 0), (0, 1))
    for row, col in walkable:
        for dr, dc in offsets:
            nr, nc = row + dr, col + dc
            if (nr, nc) not in obstacles and 0 <= nr < n_rows and 0 <= nc < n_cols:
                _add_edge(
                    graph,
                    cell_id(row, col),
                    cell_id(nr, nc),
                    kind="aisle",
                    length=cell_size,
                )

    return graph


_TYPE_COLORS: dict[str, str] = {
    "exit": "#2ca02c",
    "hallway": "#c7c7c7",
    "room": "#1f77b4",
    "stair": "#ff7f0e",
    "aisle": "#9edae5",
    "bay": "#9467bd",
}


def render_layout(
    graph: nx.Graph,
    save_path: str,
    *,
    title: str | None = None,
    dpi: int = 160,
) -> None:
    """将拓扑图画到 ``save_path``（PNG/PDF 等，由扩展名决定）。

    Parameters
    ----------
    graph :
        ``get_*_layout`` 返回的图，需含节点属性 ``pos`` 与 ``node_type``。
    save_path :
        输出路径，标量字符串。
    """
    pos: dict[Hashable, tuple[float, float]] = {
        node: (float(data["pos"][0]), float(data["pos"][1]))
        for node, data in graph.nodes(data=True)
    }
    node_colors = [
        _TYPE_COLORS.get(graph.nodes[n].get("node_type", ""), "#8c564b") for n in graph.nodes
    ]
    weights = [float(graph.nodes[n].get("vulnerable_weight", 0.0)) for n in graph.nodes]
    sizes = [420.0 + 80.0 * w for w in weights]

    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    ax_typed: Axes = ax
    nx.draw_networkx_edges(graph, pos, ax=ax_typed, edge_color="#4a4a4a", width=1.2)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax_typed,
        node_color=node_colors,
        node_size=sizes,
        edgecolors="black",
        linewidths=0.6,
    )
    nx.draw_networkx_labels(graph, pos, ax=ax_typed, font_size=7)

    layout_name = str(graph.graph.get("layout_name", "layout"))
    ax_typed.set_title(title if title is not None else layout_name)
    ax_typed.set_aspect("equal")
    ax_typed.axis("off")

    used_types: set[str] = {str(graph.nodes[n].get("node_type", "")) for n in graph.nodes}
    handles = [
        Patch(facecolor=_TYPE_COLORS[t], edgecolor="black", label=t)
        for t in sorted(used_types)
        if t in _TYPE_COLORS
    ]
    if handles:
        ax_typed.legend(handles=handles, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def searchable_nodes(graph: nx.Graph) -> list[NodeId]:
    """返回 ``search_required=True`` 的节点 id 列表。"""
    return [n for n, data in graph.nodes(data=True) if data.get("search_required")]


def exit_nodes(graph: nx.Graph) -> list[NodeId]:
    """返回所有 ``node_type==exit`` 的节点。"""
    return [n for n, data in graph.nodes(data=True) if data.get("node_type") == "exit"]


def iter_node_ids(graph: nx.Graph) -> Iterable[NodeId]:
    """按插入顺序迭代节点 id。"""
    return (str(n) for n in graph.nodes)
