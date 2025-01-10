// #pragma once 

// class AlgorithmInterface {
//  public:
//     virtual void addPoint(const void *datapoint, labeltype label, bool replace_deleted = false) = 0;

//     virtual std::priority_queue<std::pair<dist_t, labeltype>>
//         searchKnn(const void*, size_t, BaseFilterFunctor* isIdAllowed = nullptr) const = 0;

//     // Return k nearest neighbor in the order of closer fist
//     virtual std::vector<std::pair<dist_t, labeltype>>
//         searchKnnCloserFirst(const void* query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const;

//     virtual void saveIndex(const std::string &location) = 0;
//     virtual ~AlgorithmInterface(){
//     }
// };