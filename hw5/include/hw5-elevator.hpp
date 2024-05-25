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
#include <memory>
#include <set>

using namespace std;

const int NUM_FLOORS = 50;
const int NUM_ELEVATORS = 6;
const int MAX_OCCUPANCY = 10;
const int MAX_WAIT_TIME = 5000; // milliseconds

const int ASCENDING = true;
const int DESCENDING = false;

mutex mtx;
mutex global_mtx; // for information shared by both the elevator and person
mutex cout_mtx;

// Protected by mtx
queue<pair<int, int>> global_queue; // global elevator queue (floor, destination)
queue<pair<int, int>> elevator_queues[NUM_ELEVATORS];
atomic<int> num_people_serviced(0);
vector<int> elevator_positions(NUM_ELEVATORS, 0);
vector<bool> elevator_directions(NUM_ELEVATORS, true);
vector<int> global_passengers_serviced(NUM_ELEVATORS, 0);
int npeople; // global variable for the number of people to be serviced

struct Person
{
    int id;
    int curr_floor;
    int dest_floor;

    // Define operator< so that sets of persons will be ordered by destination floor.
    bool operator<(const Person& other) const
    {
        return dest_floor < other.dest_floor;
    }
};

typedef set<Person> People_t;
// Protected by global_mtx
vector<People_t> people_on_elevators(NUM_ELEVATORS);  // Set of people in each elevator
vector<People_t> people_on_floors(NUM_FLOORS);        // Set of people on each floor

bool is_person_on_floor(int person, int floor)
{
    lock_guard<mutex> lock(global_mtx);
    for(const auto& p : people_on_floors[floor])
    {
        if(p.id == person)
        {
            return true;
        }
    }
    return false;
}

bool is_stop_on_the_way(bool direction, int curr_floor, int dest_floor)
{
    if(ASCENDING == direction && dest_floor > curr_floor)
    {
        return true;
    }
    else if(DESCENDING == direction && dest_floor < curr_floor)
    {
        return true;
    }
    return false;
}

void elevator(int id) {
    int curr_floor = 0;
    int dest_floor = 0;
    int occupancy = 0;

    std::set<int> next_floors;

    while (true) {
        if(!elevator_queues[id].empty() && occupancy < MAX_OCCUPANCY)
        {
            next_floors.insert(pickup_floor);
        }
        else if(!global_queue.empty() && occupancy < MAX_OCCUPANCY)
        {
            // Peek at the next item in the global queue
            pair<int, int> next = global_queue.front();
            auto pickup_floor = next.first;
            auto direction = elevator_directions[id];
            if(is_stop_on_the_way(direction, curr_floor, pickup_floor))
            {
                // Move the request from the global queue to the elevator queue
                elevator_queues[id].push_back(next);
                global_queue.pop();
            }
        }
        else if(occupancy >= MAX_OCCUPANCY) 
        {

        }


        // if(dest_floor != curr_floor) 
        // {

        // }
        // else if(occupancy > 0) 
        // {

        // }

        // if(num_people_serviced >= npeople && elevator_queues[id].empty() && occupancy == 0) 
        // {

        // }
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

    Person me;
    me.id = id;
    me.curr_floor = curr_floor;
    me.dest_floor = dest_floor;

    global_mtx.lock();
    people_on_floors[curr_floor].insert(me); // Add person to the floor
    global_mtx.unlock();

    mtx.lock();
    global_queue.push({curr_floor, dest_floor});
    mtx.unlock();

    cout_mtx.lock();
    cout << "Person " << id << " on floor " << curr_floor << " requested elevator service to go to floor " << dest_floor << endl;
    cout_mtx.unlock();

    // Simulate moving towards the destination floor
    while (!is_person_on_floor(id, dest_floor)) {
        this_thread::sleep_for(chrono::milliseconds(1000));
    }

    cout_mtx.lock();
    cout << "Person " << id << " arrived at floor " << curr_floor << endl <<flush ;
    cout_mtx.unlock();
}


#endif // HW5_ELEVATOR_HPP

