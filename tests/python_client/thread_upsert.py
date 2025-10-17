import time
from pymilvus import MilvusClient, DataType
import random
import threading
import numpy as np
from common import common_func as cf
from common import common_type as ct


# 配置参数
nb_per_batch = 500
num_threads = 25
total_inserts = 100000  # 总共要插入的数据量


client = MilvusClient(uri="http://10.104.17.225:19530", token="root:Milvus")
dim = 768
collection_name = f"test_partial_update_perf"
# client.load_collection(collection_name)
#client.drop_collection(collection_name)
# # res = client.describe_collection(collection_name=collection_name)
# collection_name = "customized_setup_query_{}".format(int(time.time()))
schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)


# index_params = client.prepare_index_params()
# index_params.add_index(field_name="my_vector", index_type="AUTOINDEX", metric_type="COSINE")
# client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

# insert_id = 0
# insert_lock = threading.Lock()
# insert_start_time = time.time()

# def insert_loop(thread_id):
#     global insert_id
#     while True:
#         with insert_lock:
#             if insert_id >= total_inserts:
#                 break
#             start_id = insert_id
#             batch_size = min(nb_per_batch, total_inserts - insert_id)
#             insert_id += batch_size
        
#         data = cf.gen_row_data_by_schema(nb=batch_size, schema=schema, start=start_id)
#         client.insert(collection_name=collection_name, data=data)
#         print(f"[Thread-{thread_id}] Inserted {start_id} to {start_id+nb_per_batch-1}")

# insert_thread = []
# for i in range(10):
#     t = threading.Thread(target=insert_loop, args=(i,))
#     t.start()
#     insert_thread.append(t)
# for t in insert_thread:
#     t.join()

# insert_end_time = time.time()
# print(f"Insert Complete!✅")
# print(f"Insert time: {insert_end_time - insert_start_time:.2f} seconds")
# print("Now sleep for 20 min for compaction to finish")
# time.sleep(1200)
# print(f"Sleep Complete!✅")


# field 1
# client.add_collection_field(collection_name=collection_name, field_name="document", data_type=DataType.VARCHAR,
#                             nullable=True, max_length=2000)
schema.add_field(field_name="document", datatype=DataType.VARCHAR, nullable=True, max_length=2000)

# field 2
# client.add_collection_field(collection_name=collection_name, field_name="text", data_type=DataType.VARCHAR,
#                             nullable=True, max_length=2000)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, nullable=True, max_length=2000)

# field 3 
# client.add_collection_field(collection_name=collection_name, field_name="array_int", data_type=DataType.ARRAY,
#                             element_type=DataType.INT64, max_capacity=100, nullable=True)
schema.add_field(field_name="array_int", datatype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=100, nullable=True)

# field 4
# client.add_collection_field(collection_name=collection_name, field_name="array_float", data_type=DataType.ARRAY,
#                             element_type=DataType.FLOAT, max_capacity=100, nullable=True)
schema.add_field(field_name="array_float", datatype=DataType.ARRAY, element_type=DataType.FLOAT, max_capacity=100, nullable=True) 

# field 5
# client.add_collection_field(collection_name=collection_name, field_name="array_varchar", data_type=DataType.ARRAY,
#                             element_type=DataType.VARCHAR, max_capacity=100, nullable=True, max_length=200)
schema.add_field(field_name="array_varchar", datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=100, nullable=True, max_length=200)

# field 6
# client.add_collection_field(collection_name=collection_name, field_name="array_bool", data_type=DataType.ARRAY,
#                             element_type=DataType.BOOL, max_capacity=100, nullable=True)
schema.add_field(field_name="array_bool", datatype=DataType.ARRAY, element_type=DataType.BOOL, max_capacity=100, nullable=True)

# field 7
# client.add_collection_field(collection_name=collection_name, field_name="json_field", data_type=DataType.JSON,
#                             nullable=True)
schema.add_field(field_name="json_field", datatype=DataType.JSON, nullable=True)

# field 8
# client.add_collection_field(collection_name=collection_name, field_name="float_field", data_type=DataType.FLOAT,
#                             nullable=True)
schema.add_field(field_name="float_field", datatype=DataType.FLOAT, nullable=True)

# field 9
# client.add_collection_field(collection_name=collection_name, field_name="int32_field", data_type=DataType.INT32, nullable=True)
schema.add_field(field_name="int32_field", datatype=DataType.INT32, nullable=True)

# field 10
# client.add_collection_field(collection_name=collection_name, field_name="bool_field", data_type=DataType.BOOL, nullable=True)
schema.add_field(field_name="bool_field", datatype=DataType.BOOL, nullable=True)

print(f"Add collection field Complete!✅")


# 统计变量
id_counter = 0
id_lock = threading.Lock()
completed_upserts = 0
error_count = 0
total_latency = 0.0
max_latency = 0.0
start_time = time.time()

# 并发写入函数
def upsert_loop(thread_id):
    global id_counter, completed_upserts, error_count, total_latency, max_latency

    while True:
        with id_lock:
            if id_counter >= total_inserts:
                break
            start_id = id_counter
            batch_size = min(nb_per_batch, total_inserts - id_counter)
            id_counter += batch_size

        try:
            #print(f"[Thread-{thread_id}] start upsert {start_id} to {start_id+batch_size-1}")
            batch_start_time = time.time()
            data = cf.gen_row_data_by_schema(nb=batch_size, schema=schema, start=start_id, skip_field_names=["my_vector"])
            client.upsert(collection_name, data, partial_update=True)
            print(f"[Thread-{thread_id}] Upserted {start_id} to {start_id+batch_size-1}")
            with id_lock:
                completed_upserts += batch_size

            latency = time.time() - batch_start_time
            with id_lock:
                total_latency += latency
                max_latency = max(max_latency, latency)
        except Exception as e:
            with id_lock:
                error_count += batch_size
            print(f"[Thread-{thread_id}] Operation error: {e}")

       
# 启动多个线程
threads = []
print(f"Starting {num_threads} threads")
for t_id in range(num_threads):
    t = threading.Thread(target=upsert_loop, args=(t_id,))
    t.start()
    threads.append(t)

# 等待所有线程完成
for t in threads:
    t.join()


# 计算并输出最终统计结果
elapsed_time = time.time() - start_time
total_operations = completed_upserts
qps = total_operations / elapsed_time if elapsed_time > 0 else 0
avg_latency = total_latency / (total_operations / nb_per_batch) if total_operations > 0 else 0

print("\n=== 执行结果统计 ===")
print(f"总更新数量: {completed_upserts}")
print(f"总耗时: {elapsed_time:.2f}秒")
print(f"QPS: {qps:.1f}")
print(f"平均延迟: {avg_latency:.3f}秒")
print(f"最大延迟: {max_latency:.3f}秒")
print(f"错误数量: {error_count}")
print(f"错误率: {error_count/total_inserts*100:.2f}%")

#client.release_collection(collection_name)
#print(f"Release collection Complete!✅")