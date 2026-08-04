"""第 28 课 RAGAS 评测 — 10 组 QA 测试数据集。(第31课引用)

覆盖场景：
  Q1~Q2: 完美检索 + 忠实回答（高分基准）
  Q3~Q4: 答案幻觉 / 不切题（低分对照）
  Q5~Q6: 检索缺失 / 噪声上下文
  Q7~Q8: 极端输入（空 context、极短回答）
  Q9~Q10: 多段落上下文 + 部分相关
"""
from my_math.ragas_eval import RAGSample


SAMPLES = [
    # ── 高分基准 ──
    RAGSample(
        question="What is ROS2?",
        contexts=[
            "ROS2 is the next generation of Robot Operating System.",
            "It uses DDS middleware for communication between nodes.",
        ],
        answer="ROS2 is a next-generation robot operating system that uses DDS for communication.",
        ground_truth="ROS2 is a robot operating system for developing robot applications with DDS.",
    ),
    RAGSample(
        question="What is a costmap in Nav2?",
        contexts=[
            "A costmap is a 2D grid where each cell has a cost value from 0 to 254.",
            "Nav2 uses global and local costmaps for path planning.",
        ],
        answer="In Nav2, a costmap is a 2D grid with values 0-254 used for path planning.",
        ground_truth="A costmap is a 2D occupancy grid used in Nav2 for navigation planning.",
    ),

    # ── 低分对照：幻觉 ──
    RAGSample(
        question="How does Nav2 handle obstacle avoidance?",
        contexts=[
            "Nav2 uses a local costmap with sensor data to detect obstacles.",
        ],
        answer="Nav2 uses deep reinforcement learning to train robots to avoid obstacles and fly drones.",
        ground_truth="Nav2 uses local costmaps and controller plugins to avoid obstacles.",
    ),
    RAGSample(
        question="What is the purpose of AMCL?",
        contexts=[
            "AMCL stands for Adaptive Monte Carlo Localization.",
            "It uses particle filters to estimate robot pose from laser scans and odometry.",
        ],
        answer="AMCL is used for mapping the environment and building 3D models.",
        ground_truth="AMCL is a localization algorithm that estimates robot position using particle filters.",
    ),

    # ── 检索缺失 / 噪声 ──
    RAGSample(
        question="What is DDS?",
        contexts=[
            "The weather today is sunny with a high of 25 degrees.",
            "Many people enjoy outdoor activities on sunny days.",
        ],
        answer="I cannot answer the question based on the provided context.",
        ground_truth="DDS is Data Distribution Service, a middleware protocol for real-time systems.",
    ),
    RAGSample(
        question="How to configure Nav2 plugins?",
        contexts=[
            "Nav2 plugins are configured through YAML parameter files.",
            "The default parameters are in nav2_params.yaml under the nav2_bringup package.",
            "Unrelated: Python is a versatile programming language.",
        ],
        answer="Nav2 plugins use YAML configuration files with parameters specified in nav2_params.yaml.",
        ground_truth="Nav2 plugins can be configured via YAML files and loaded through behavior tree XML.",
    ),

    # ── 极端输入 ──
    RAGSample(
        question="What is SLAM?",
        contexts=[],
        answer="SLAM stands for Simultaneous Localization and Mapping.",
        ground_truth="SLAM constructs a map while keeping track of the robot's location.",
    ),
    RAGSample(
        question="Is Nav2 open source?",
        contexts=["Yes, Nav2 is open source under the Apache 2.0 license."],
        answer="Yes.",
        ground_truth="Nav2 is open source under Apache 2.0 license.",
    ),

    # ── 多段落 + 部分相关 ──
    RAGSample(
        question="How does behavior tree work in Nav2?",
        contexts=[
            "Behavior trees control the flow of robot tasks in Nav2.",
            "They use XML format to define sequences, fallbacks, and conditions.",
            "Groot is a visual editor for behavior trees.",
        ],
        answer="Behavior trees in Nav2 define task flow using XML with sequences, fallbacks, and conditions.",
        ground_truth="Behavior trees organize navigation tasks as tree nodes executing in priority order.",
    ),
    RAGSample(
        question="What sensors does Nav2 support?",
        contexts=[
            "Nav2 supports LiDAR sensors for obstacle detection.",
            "RGB-D cameras can also be used for perception.",
            "Support for IMU and odometry sensors is included.",
            "The Linux kernel version 5.15 includes new networking features.",
        ],
        answer="Nav2 supports LiDAR, cameras, IMU, and odometry sensors.",
        ground_truth="Nav2 supports LiDAR, RGB-D cameras, IMU, and odometry sensors for navigation.",
    ),
]
