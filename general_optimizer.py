import os
import tensorflow as tf

# Runtime optimizations
def apply_runtime_optimizations():
    logger.info("Applying runtime optimizations")

    total_cpu_cores = os.popen('nproc').read().strip()
    number_sockets = int(os.popen('grep "^physical id" /proc/cpuinfo | awk \'{print $4}\' | sort -un | tail -1').read().strip()) + 1
    number_cpu_cores = int((int(total_cpu_cores) / 2) / number_sockets)

    logger.info("number of CPU cores per socket: {}".format(number_cpu_cores))
    logger.info("number of sockets: {}".format(number_sockets))

    # set intra_op_parallelism = number of physical core per socket
    # set inter_op_parallelism = number of sockets

    tf.config.set_soft_device_placement(True)
    tf.config.threading.set_intra_op_parallelism_threads(number_cpu_cores)
    tf.config.threading.set_inter_op_parallelism_threads(number_sockets)

# Graph optimizations
def apply_graph_optimizations():
    logger.info("Applying graph optimizations")

    # Export the graph to a protobuf file
    tf.io.write_graph(tf.compat.v1.get_default_graph(), './', 'graph.pbtxt', as_text=True)

    # Freeze the graph
    os.system('python3 -m tensorflow.python.tools.freeze_graph --input_graph crypto_spider_5g_fcnn_frozen.pb --input_checkpoint crypto_spider_5g_fcnn.h5 --output_graph crypto_spider_5g_fcnn_frozen.pb --output_node_names model_output')

    # Optimize the graph for inference
    os.system('python3 -m tensorflow.python.tools.optimize_for_inference --input crypto_spider_5g_fcnn_frozen.pb --output crypto_spider_5g_fcnn_frozen.pb --input_names input_1 --output_names model_output --frozen_func true')