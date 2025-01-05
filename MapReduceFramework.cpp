#include <csignal>
#include <atomic>
#include "algorithm"
#include <thread>
#include <iostream>
#include "vector"
#include "MapReduceFramework.h"
#include "MapReduceClient.h"
#include "Barrier.h"
#include "Barrier.cpp"

using std::vector;

struct SharedThreadContext;
struct ThreadContext;
void free_all (JobHandle job);
K2 *find_max_key (std::vector<IntermediateVec *> *intermediate_vecs);
void create_threads (SharedThreadContext *);
void map_phase (ThreadContext *tc);
void sort_phase (ThreadContext *tc);
void shuffle_phase (ThreadContext *tc);
void reduce_phase (ThreadContext *tc);
void exit_in_error (void *)
{
  std::cout << "system error: pthread library or mutex library returned un "
               "error\n" << std::endl;
  exit (1);
}

struct SharedThreadContext
{
    const MapReduceClient *map_reduce_client;
    const InputVec *input_vec;
    OutputVec *output_vec;
    int multiThreadLevel;
    Barrier *barrier;
    vector<IntermediateVec *> k2_vecs_before_shuffle;
    vector<IntermediateVec *> k2_vecs_after_shuffle;
    std::atomic<long unsigned int> *counter_index_input_vector;
    std::atomic<int> *counter_map_job_done;
    std::atomic<long unsigned int> *counter_index_intermediate_vec;
    std::atomic<int> *counter_reduce_job_done;
    pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
    stage_t stage = UNDEFINED_STAGE;
    int counter_shuffle_job_total;
    int counter_shuffle_job_done;
    pthread_t *threads_id;
    ThreadContext **thread_contexts;

};

struct ThreadContext
{
    int tid;
    SharedThreadContext *shared_thread_context;
};

struct PrivateJobHandler
{
    SharedThreadContext *shared_thread_context;
    bool wait_for_job_called = false;
};

void getJobState (JobHandle job, JobState *state)
{

  PrivateJobHandler *private_job_handler = static_cast<PrivateJobHandler *>
  (job);
  SharedThreadContext *shared_thread_context =
      private_job_handler->shared_thread_context;
  stage_t stage = shared_thread_context->stage;
  state->stage = stage;
  if (stage == UNDEFINED_STAGE)
  {
    state->percentage = 0;
    return;
  }
  if (stage == MAP_STAGE)
  {
    int job_done = shared_thread_context->counter_map_job_done->load ();
    int total_job = shared_thread_context->input_vec->size ();
    if (total_job == 0) state->percentage = 0;
    else state->percentage = (((float) job_done) / total_job) * 100;
    return;
  }
  if (stage == SHUFFLE_STAGE)
  {
    int job_done = shared_thread_context->counter_shuffle_job_done;
    int total_job = shared_thread_context->counter_shuffle_job_total;
    if (total_job == 0) state->percentage = 0;
    else state->percentage = (((float) job_done) / total_job) * 100;
    return;
  }
  if (stage == REDUCE_STAGE)
  {
    int job_done = shared_thread_context->counter_reduce_job_done->load ();
    int total_job = shared_thread_context->counter_shuffle_job_total;
    if (total_job == 0) state->percentage = 0;
    else state->percentage = (((float) job_done) / total_job) * 100;
  }
}

void *start_point (void *arg)
{
  ThreadContext *tc = (ThreadContext *) arg;
  if (tc->tid == 0)
  {
    tc->shared_thread_context->stage = MAP_STAGE;
  }
  tc->shared_thread_context->barrier->barrier ();
  map_phase (tc);
  sort_phase (tc);
  tc->shared_thread_context->barrier->barrier ();
  if (tc->tid == 0)
  {
    tc->shared_thread_context->stage = SHUFFLE_STAGE;
    shuffle_phase (tc);
    tc->shared_thread_context->stage = REDUCE_STAGE;
  }
  tc->shared_thread_context->barrier->barrier ();
  reduce_phase (tc);
  return nullptr;
}

void waitForJob (JobHandle job)
{
  PrivateJobHandler *private_job_handler = (PrivateJobHandler *) job;
  if (private_job_handler->wait_for_job_called) return;
  pthread_t *threads = private_job_handler->shared_thread_context->threads_id;
  for (int i = 0; i < private_job_handler->shared_thread_context
      ->multiThreadLevel; i++)
  {
    int status = pthread_join ((threads[i]), NULL);
    if (status != 0) exit_in_error (nullptr);
  }

}

void closeJobHandle (JobHandle job)
{
  waitForJob (job);
  free_all (job);
}

JobHandle startMapReduceJob (const MapReduceClient &client,
                             const InputVec &inputVec, OutputVec &outputVec,
                             int multiThreadLevel)
{
  SharedThreadContext *shared_thread_context = new SharedThreadContext ();
  shared_thread_context->map_reduce_client = &client;
  shared_thread_context->input_vec = &inputVec;
  shared_thread_context->output_vec = &outputVec;
  shared_thread_context->multiThreadLevel = multiThreadLevel;
  shared_thread_context->barrier = new Barrier (multiThreadLevel);
  shared_thread_context->counter_index_input_vector = new std::atomic<long unsigned int> (0);
  shared_thread_context->counter_index_intermediate_vec = new std::atomic<long unsigned int> (0);
  shared_thread_context->counter_reduce_job_done = new std::atomic<int> (0);
  shared_thread_context->counter_map_job_done = new std::atomic<int> (0);
  for (int i = 0; i < multiThreadLevel; i++)
    shared_thread_context->k2_vecs_before_shuffle.push_back (new IntermediateVec);
  shared_thread_context->counter_shuffle_job_total = 0;
  shared_thread_context->counter_shuffle_job_done = 0;

  create_threads (shared_thread_context);

  PrivateJobHandler *job_handler = new PrivateJobHandler ();
  job_handler->wait_for_job_called = false;
  job_handler->shared_thread_context = shared_thread_context;
  return job_handler;

}

void create_threads (SharedThreadContext *shared_thread_context)
{
  int num_of_threads = shared_thread_context->multiThreadLevel;

  pthread_t *threads_id = new pthread_t[num_of_threads];
  ThreadContext **thread_contexts = new ThreadContext *[num_of_threads];
  shared_thread_context->threads_id = threads_id;
  shared_thread_context->thread_contexts = thread_contexts;

  for (int i = 0; i < num_of_threads; ++i)
  {
    thread_contexts[i] = new ThreadContext{i, shared_thread_context};
  }

  for (int i = 0; i < num_of_threads; ++i)
  {
    int status = pthread_create (threads_id + i,
                                 NULL, start_point, thread_contexts[i]);
    if (status != 0) exit_in_error (nullptr);
  }

}

void map_phase (ThreadContext *tc)
{
  SharedThreadContext *shared_thread_context = tc->shared_thread_context;

  const InputVec *input_vec = shared_thread_context->input_vec;
  long unsigned int old_value = (*(shared_thread_context->counter_index_input_vector))++;

  while (old_value < (*input_vec).size ())
  {

    shared_thread_context->map_reduce_client->map (
        (*input_vec)[old_value].first,
        (*input_vec)[old_value].second,
        shared_thread_context->k2_vecs_before_shuffle[tc->tid]);

    (*(shared_thread_context->counter_map_job_done))++;
    old_value = (*(shared_thread_context->counter_index_input_vector))++;
  }

}

void sort_phase (ThreadContext *tc)
{
  std::sort (tc->shared_thread_context->k2_vecs_before_shuffle[tc->tid]->begin (),
             tc->shared_thread_context->k2_vecs_before_shuffle[tc->tid]->end (),
             [] (const std::pair<K2 *, V2 *> &a, const std::pair<K2 *, V2 *> &b)
             {
                 return *a.first < *b.first;
             });

}

void reduce_phase (ThreadContext *tc)
{
  SharedThreadContext *shared_thread_context = tc->shared_thread_context;

  long unsigned int old_value = (*
      (shared_thread_context->counter_index_intermediate_vec))++;

  while (old_value < shared_thread_context->k2_vecs_after_shuffle.size ())
  {
    IntermediateVec *intermediate_vec =
        shared_thread_context->k2_vecs_after_shuffle[old_value];

    shared_thread_context->map_reduce_client->reduce (intermediate_vec,
                                                      shared_thread_context);

    int vec_size = (*intermediate_vec).size ();
    (*(shared_thread_context->counter_reduce_job_done)) += vec_size;
    old_value = (*(shared_thread_context->counter_index_intermediate_vec))++;
  }

}

void emit2 (K2 *key, V2 *value, void *context)
{

  IntermediateVec *intermediate_vec = (IntermediateVec *) context;
  intermediate_vec->push_back (IntermediatePair (key, value));

}

void emit3 (K3 *key, V3 *value, void *context)
{
  SharedThreadContext *shared_thread_context = (SharedThreadContext *) context;

  int status = pthread_mutex_lock (&shared_thread_context->mtx);
  if (status != 0) exit_in_error (nullptr);
  shared_thread_context->output_vec->push_back (OutputPair (key, value));
  status = pthread_mutex_unlock (&shared_thread_context->mtx);
  if (status != 0) exit_in_error (nullptr);
}

void pop_and_add_max_key (K2 *key, IntermediateVec *vec_for_key,
                          std::vector<IntermediateVec *> *intermediate_vecs,
                          int &counter);

void shuffle_phase (ThreadContext *tc)
{

  for (auto vec: tc->shared_thread_context->k2_vecs_before_shuffle)
  {
    tc->shared_thread_context->counter_shuffle_job_total += vec->size ();
  }

  K2 *max_key = find_max_key (&tc->shared_thread_context->k2_vecs_before_shuffle);

  while (max_key != nullptr)
  {
    IntermediateVec *vec_for_key = new IntermediateVec;
    pop_and_add_max_key (max_key, vec_for_key,
                         &tc->shared_thread_context->k2_vecs_before_shuffle,
                         tc->shared_thread_context->counter_shuffle_job_done);
    tc->shared_thread_context->k2_vecs_after_shuffle.push_back (vec_for_key);
    max_key = find_max_key (&tc->shared_thread_context->k2_vecs_before_shuffle);
  }

}

void pop_and_add_max_key (K2 *key, IntermediateVec *vec_for_key,
                          std::vector<IntermediateVec *> *intermediate_vecs,
                          int &counter)
{
  for (IntermediateVec *vec: *intermediate_vecs)
  {
    IntermediatePair last_pair;
    while (vec != nullptr && !vec->empty ())
    {
      last_pair = vec->back ();
      if (*last_pair.first < *key) break;

      vec->pop_back ();
      vec_for_key->push_back (last_pair);
      counter++;
    }
  }
}

K2 *find_max_key (std::vector<IntermediateVec *> *intermediate_vecs)
{
  K2 *max_key = nullptr;
  for (IntermediateVec *vec: *intermediate_vecs)
  {
    if (vec != nullptr && vec->empty () == false)
    {
      if (max_key == nullptr or *max_key < *vec->back ().first)
      {
        max_key = vec->back ().first;
      }
    }
  }
  return max_key;
}

void free_all (JobHandle job)
{
  PrivateJobHandler *private_job_handler = (PrivateJobHandler *) job;
  SharedThreadContext *shared_thread_context =
      private_job_handler->shared_thread_context;
  int status = pthread_mutex_destroy (&shared_thread_context->mtx);
  if (status != 0) exit_in_error (nullptr);
  delete shared_thread_context->barrier;
  delete shared_thread_context->counter_index_input_vector;
  delete shared_thread_context->counter_index_intermediate_vec;
  delete shared_thread_context->counter_reduce_job_done;
  delete shared_thread_context->counter_map_job_done;
  delete shared_thread_context->threads_id;
  for (int i = 0; i < shared_thread_context->multiThreadLevel; i++)
  {
    delete shared_thread_context->k2_vecs_before_shuffle[i];
    delete shared_thread_context->thread_contexts[i];
  }
  delete shared_thread_context->thread_contexts;
  for (auto vec: shared_thread_context->k2_vecs_after_shuffle)
    delete vec;
  delete private_job_handler;
  delete shared_thread_context;
}




