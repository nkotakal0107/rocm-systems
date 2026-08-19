/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// Pins the SMN loop-back count of the reg-state parser: N register entries must
// yield exactly N passes. Images are sized exactly, so an extra pass overruns
// the read length and the parser returns -1. No GPU required.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rocm_smi/rocm_smi.h"

namespace amd::smi {
int present_reg_state(const char* fname, rsmi_reg_type_t reg_type, rsmi_name_value_t** kv,
                      uint32_t* kvnum);
}  // namespace amd::smi

namespace {

// Mirrors xgmi_regs[]: 8B header, 8B instance header, then 16B per SMN entry.
std::string WriteXgmiImage(uint16_t num_smn_regs) {
  const size_t size = 16 + static_cast<size_t>(num_smn_regs) * 16;
  std::vector<uint8_t> buf(size, 0);

  const uint16_t structure_size = static_cast<uint16_t>(size);
  std::memcpy(&buf[0], &structure_size, sizeof(structure_size));
  buf[5] = 1;  // num_instances
  std::memcpy(&buf[12], &num_smn_regs, sizeof(num_smn_regs));

  for (uint16_t i = 0; i < num_smn_regs; ++i) {
    const size_t offset = 16 + static_cast<size_t>(i) * 16;
    const uint64_t addr = static_cast<uint64_t>(i) + 1;
    const uint32_t value = static_cast<uint32_t>(i) * 10;
    std::memcpy(&buf[offset], &addr, sizeof(addr));
    std::memcpy(&buf[offset + 8], &value, sizeof(value));
  }

  char path[] = "/tmp/amdsmi_xgmi_regs_XXXXXX";
  const int fd = mkstemp(path);
  EXPECT_GE(fd, 0);
  EXPECT_EQ(write(fd, buf.data(), size), static_cast<ssize_t>(size));
  close(fd);
  return std::string(path);
}

// 6 header fields + 4 instance-header fields + 3 fields per SMN entry.
uint32_t ExpectedPairs(uint16_t num_smn_regs) {
  return 10U + 3U * static_cast<uint32_t>(num_smn_regs);
}

void ExpectSmnPasses(uint16_t num_smn_regs) {
  const std::string path = WriteXgmiImage(num_smn_regs);
  rsmi_name_value_t* kv = nullptr;
  uint32_t kvnum = 0;

  EXPECT_EQ(amd::smi::present_reg_state(path.c_str(), RSMI_REG_XGMI, &kv, &kvnum), 0);
  EXPECT_EQ(kvnum, ExpectedPairs(num_smn_regs));

  free(kv);
  remove(path.c_str());
}

TEST(GpuUnit, BinaryParserXgmiOneSmnReg) { ExpectSmnPasses(1); }

TEST(GpuUnit, BinaryParserXgmiTwoSmnRegs) { ExpectSmnPasses(2); }

TEST(GpuUnit, BinaryParserXgmiThreeSmnRegs) { ExpectSmnPasses(3); }

}  // namespace
