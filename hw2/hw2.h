#pragma once
#include <cmath>
#include <iostream>
#include <math.h>

/**
 * Measure the precision of data_type on this machine.
 * 
 * data_type should be either float or double.
 *
 * @param pow_func Pointer to an appropriate power function to use for 
 *                 data_type.  For float, use powf().  For double, use pow().
 */
template<class data_type>
data_type measure_precision(data_type (*pow_func)(data_type, data_type))
{
    data_type y, prev_y;
    int j = 1;

    // Compute 1 - (1 + 1 / 2^j) in a loop.  The result will approach zero
    // as j increases.  When the result compares equal to zero, then we have
    // reached the precision limit of the machine.
    do
    {
        prev_y = y;
        y = static_cast<data_type>(1) - 
            (static_cast<data_type>(1) + static_cast<data_type>(1)
             / pow_func(2, j));
        j++;
        if(!j%1000)
        {
            // Print out the count so we know the program is still running
            std::cout << "Count: " << j << " y: " << y << std::endl;
        }
    }
    while( y != 0 );

    // std::cout << "Number of iterations to converge: " << j << std::endl;

    // The precision equals prev_y - y.  However, y tested equal to 0, so the
    // precision is just abs(prev_y).
    return abs(prev_y);
}

template<class data_type>
data_type multiply_as_type()
{
    data_type result = static_cast<data_type>(200) *
                       static_cast<data_type>(300) *
                       static_cast<data_type>(400) *
                       static_cast<data_type>(500);
    return result;
}