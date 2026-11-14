#pragma once
/// @file Status.h  Result codes returned by all fallible TinyYolo functions.

#include <stdint.h>

namespace tinyyolo {

enum class Status : uint8_t {
  Ok = 0,
  InvalidArgument,  ///< A parameter is null or out of range.
  InvalidModel,     ///< The network description is inconsistent.
  ShapeMismatch,    ///< Tensor shapes do not fit the operation.
  OutOfMemory,      ///< An allocation failed.
  WeightError,      ///< The weight source is missing or too short.
  NotInitialized,   ///< begin() was not called successfully.
  Unsupported       ///< The requested feature is not available.
};

inline const char* statusName(Status status) {
  switch (status) {
    case Status::Ok: return "Ok";
    case Status::InvalidArgument: return "InvalidArgument";
    case Status::InvalidModel: return "InvalidModel";
    case Status::ShapeMismatch: return "ShapeMismatch";
    case Status::OutOfMemory: return "OutOfMemory";
    case Status::WeightError: return "WeightError";
    case Status::NotInitialized: return "NotInitialized";
    case Status::Unsupported: return "Unsupported";
  }
  return "Unknown";
}

}  // namespace tinyyolo

/// Returns the status from the enclosing function when it is not Ok.
/// The failing function is responsible for logging the reason.
#define TINYYOLO_CHECK(expr)                                              \
  do {                                                                    \
    const ::tinyyolo::Status tinyyolo_status_ = (expr);                   \
    if (tinyyolo_status_ != ::tinyyolo::Status::Ok) return tinyyolo_status_; \
  } while (0)
