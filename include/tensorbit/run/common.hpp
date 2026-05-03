#ifndef TENSORBIT_RUN_COMMON_HPP
#define TENSORBIT_RUN_COMMON_HPP

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <format>
#include <memory>
#include <mutex>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

namespace tensorbit {
namespace run {

/* ================================================================
 * Unexpected<E> — must be defined BEFORE Result<T,E>
 * ================================================================ */

template <typename E>
class Unexpected {
public:
    explicit Unexpected(E e) : error_(std::move(e)) {}
    const E& value() const { return error_; }
    E&       value() { return error_; }

private:
    E error_;
};

template <typename E>
Unexpected<typename std::decay<E>::type> unexpected(E&& e) {
    return Unexpected<typename std::decay<E>::type>(std::forward<E>(e));
}

/* ================================================================
 * Result<T,E> — C++20 std::expected replacement
 * ================================================================ */

template <typename T, typename E>
class Result {
public:
    Result(T value) : data_(std::move(value)) {}
    Result(Unexpected<E>&& unex) : data_(std::move(unex.value())) {}

    bool has_value() const { return data_.index() == 0; }
    explicit operator bool() const { return has_value(); }

    T&       value() { return std::get<0>(data_); }
    const T& value() const { return std::get<0>(data_); }
    T&       operator*() { return value(); }
    const T& operator*() const { return value(); }

    E&       error() { return std::get<1>(data_); }
    const E& error() const { return std::get<1>(data_); }

private:
    std::variant<T, E> data_;
};

/* ================================================================
 * Result<void,E> specialization
 * ================================================================ */

template <typename E>
class Result<void, E> {
public:
    Result() : data_() {}
    Result(Unexpected<E>&& unex) : data_(std::move(unex.value())) {}

    bool has_value() const { return data_.index() == 0; }
    explicit operator bool() const { return has_value(); }

    void value() const {}
    E&       error() { return std::get<1>(data_); }
    const E& error() const { return std::get<1>(data_); }

private:
    /* std::monostate for 'void' value, E for error */
    std::variant<std::monostate, E> data_;
};

/* ================================================================
 * Logger — thread-safe singleton, matches tensorbit-core pattern
 * ================================================================ */

enum class LogLevel {
    kTrace,
    kDebug,
    kInfo,
    kWarn,
    kError,
    kFatal,
};

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    void log(LogLevel level, std::string_view msg,
             const std::source_location& loc = std::source_location::current()) {
        std::lock_guard<std::mutex> lock(mutex_);
        const char*                 prefix = level_prefix(level);
        auto                        now = std::chrono::system_clock::now();
        std::time_t                 tt = std::chrono::system_clock::to_time_t(now);
        struct tm                   tm_buf;
#ifdef _WIN32
        localtime_s(&tm_buf, &tt);
#else
        localtime_r(&tt, &tm_buf);
#endif
        std::fprintf(stderr, "[%02d:%02d:%02d] %-5s %s:%u %s\n", tm_buf.tm_hour, tm_buf.tm_min,
                     tm_buf.tm_sec, prefix, loc.file_name(), loc.line(), msg.data());
    }

private:
    Logger() = default;
    std::mutex mutex_;

    static const char* level_prefix(LogLevel level) {
        switch (level) {
            case LogLevel::kTrace:
                return "TRACE";
            case LogLevel::kDebug:
                return "DEBUG";
            case LogLevel::kInfo:
                return "INFO";
            case LogLevel::kWarn:
                return "WARN";
            case LogLevel::kError:
                return "ERROR";
            case LogLevel::kFatal:
                return "FATAL";
            default:
                return "????";
        }
    }
};

}  // namespace run
}  // namespace tensorbit

/* ================================================================
 * Logging macros (C++20, uses vformat + make_format_args)
 * CRITICAL: All format args MUST be lvalues (named variables).
 * Undefine C-level macros to avoid conflicts.
 * ================================================================ */

#ifdef TB_LOG_ERROR
#undef TB_LOG_ERROR
#endif
#ifdef TB_LOG_WARN
#undef TB_LOG_WARN
#endif
#ifdef TB_LOG_INFO
#undef TB_LOG_INFO
#endif
#ifdef TB_LOG_DEBUG
#undef TB_LOG_DEBUG
#endif

#define TB_LOG_TRACE(fmt, ...)                                                        \
    do {                                                                              \
        auto msg = std::vformat((fmt), std::make_format_args(__VA_ARGS__));          \
        ::tensorbit::run::Logger::instance().log(::tensorbit::run::LogLevel::kTrace, msg); \
    } while (0)

#define TB_LOG_DEBUG(fmt, ...)                                                        \
    do {                                                                              \
        auto msg = std::vformat((fmt), std::make_format_args(__VA_ARGS__));          \
        ::tensorbit::run::Logger::instance().log(::tensorbit::run::LogLevel::kDebug, msg); \
    } while (0)

#define TB_LOG_INFO(fmt, ...)                                                         \
    do {                                                                              \
        auto msg = std::vformat((fmt), std::make_format_args(__VA_ARGS__));          \
        ::tensorbit::run::Logger::instance().log(::tensorbit::run::LogLevel::kInfo, msg);  \
    } while (0)

#define TB_LOG_WARN(fmt, ...)                                                         \
    do {                                                                              \
        auto msg = std::vformat((fmt), std::make_format_args(__VA_ARGS__));          \
        ::tensorbit::run::Logger::instance().log(::tensorbit::run::LogLevel::kWarn, msg);  \
    } while (0)

#define TB_LOG_ERROR(fmt, ...)                                                        \
    do {                                                                              \
        auto msg = std::vformat((fmt), std::make_format_args(__VA_ARGS__));          \
        ::tensorbit::run::Logger::instance().log(::tensorbit::run::LogLevel::kError, msg); \
    } while (0)

#endif /* TENSORBIT_RUN_COMMON_HPP */
