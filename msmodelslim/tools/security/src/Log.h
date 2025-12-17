/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CORE_LOG_H
#define CORE_LOG_H

#include <string>
#include <map>
#include <memory>
#include <ostream>
#include <iostream>

#define FILELINE __FILE__, __FUNCTION__, __LINE__

constexpr int LOG_DEBUG_LEVEL = 1;
constexpr int LOG_INFO_LEVEL = 2;
constexpr int LOG_WARNING_LEVEL = 3;
constexpr int LOG_ERROR_LEVEL = 4;

extern int g_frizyLogLevel;

namespace Base {
void SETLOGLEVEL(int level);
}


#define DEBUG_LOG(fmt, args...)  do { if (g_frizyLogLevel <= LOG_DEBUG_LEVEL) \
    { printf("[DEBUG] " fmt "\n", ##args); fflush(stdout); } } while (0)
#define INFO_LOG(fmt, args...)  do { if (g_frizyLogLevel <= LOG_INFO_LEVEL) \
    { printf("[INFO] " fmt "\n", ##args); fflush(stdout); } } while (0)
#define WARN_LOG(fmt, args...)  do { if (g_frizyLogLevel <= LOG_WARNING_LEVEL) \
    { printf("[WARN] " fmt "\n", ##args); fflush(stdout); } } while (0)
#define ERROR_LOG(fmt, args...)  do { if (g_frizyLogLevel <= LOG_ERROR_LEVEL) \
    { printf("[ERROR] " fmt "\n", ##args); fflush(stdout); } } while (0)
#define PROMPT_MSG(fmt, args...) printf(fmt, ##args)

#endif  // CORE_LOG_H