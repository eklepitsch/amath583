// starter code for HW5 - elevator problem
#ifndef HW5_ELEVATOR_HPP
#define HW5_ELEVATOR_HPP

#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <chrono>
#include <random>
#include <atomic>
#include <vector>
#include <limits>

using namespace std;

const int NUM_FLOORS = 50;
const int NUM_ELEVATORS = 6;
const int MAX_OCCUPANCY = 10;
const int MAX_WAIT_TIME = 5000; // milliseconds

mutex mtx;
mutex global_mtx; // for information shared by both the elevator and person
mutex cout_mtx;
queue<pair<int, int>> global_queue; // global elevator queue (floor, destination)
queue<pair<int, int>> elevator_queues[NUM_ELEVATORS];
atomic<int> num_people_serviced(0);
vector<int> elevator_positions(NUM_ELEVATORS, 0);
vector<bool> elevator_directions(NUM_ELEVATORS, true);
vector<int> global_passengers_serviced(NUM_ELEVATORS, 0);
int npeople; // global variable for the number of people to be serviced

void elevator(int id) {
    int curr_floor = 0;
    int dest_floor = 0;
    int occupancy = 0;

    while (true) {
    // (!elevator_queues[id].empty() && occupancy < MAX_OCCUPANCY)
    // else (!global_queue.empty() && occupancy < MAX_OCCUPANCY)
    // else (occupancy >= MAX_OCCUPANCY) 


    // (dest_floor != curr_floor) 
    // else (occupancy > 0) 

    // (num_people_serviced >= npeople && elevator_queues[id].empty() && occupancy == 0) 

    }
}

void person(int id) {
    static atomic<int> counter(0); // static counter for unique IDs
    int person_id = counter.fetch_add(1); // Increment and assign ID atomically

    int curr_floor = rand() % NUM_FLOORS;
    int dest_floor = rand() % NUM_FLOORS;
    while (dest_floor == curr_floor) {
        dest_floor = rand() % NUM_FLOORS;
    }

    cout_mtx.lock();
    cout << "Person " << id << " wants to go from floor " << curr_floor << " to floor " << dest_floor << endl;
    cout_mtx.unlock();

    mtx.lock();
    global_queue.push({curr_floor, dest_floor});
    mtx.unlock();

    cout_mtx.lock();
    cout << "Person " << id << " on floor " << curr_floor << " requested elevator service to go to floor " << dest_floor << endl;
    cout_mtx.unlock();

    // Simulate moving towards the destination floor
    while (curr_floor != dest_floor) {
        this_thread::sleep_for(chrono::milliseconds(1000));
    }

    cout_mtx.lock();
    cout << "Person " << id << " arrived at floor " << curr_floor << endl <<flush ;
    cout_mtx.unlock();
}


#endif // HW5_ELEVATOR_HPP

