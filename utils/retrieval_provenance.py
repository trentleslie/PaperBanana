# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The vocabulary for what retrieval actually did, shared by writer and reader.

The mode a run *asked for* and the mode it *ran in* are two different facts:
``RetrieverAgent`` downgrades ``auto`` and ``random`` to ``none`` when the
reference file is absent, and ``manual`` likewise when its selection file is.
Recording only the request let a manifest assert ``"auto"`` for a run that
retrieved nothing, which is exactly what every run to date has done.

The agent writes the effective mode onto the data dict under
``EFFECTIVE_RETRIEVAL_KEY``; the manifest reads it back rather than re-deriving
the downgrade from the same filesystem state at a later, different moment. This
module exists so the two ends cannot drift apart, and so it can be imported
without pulling in the provider SDKs that ``agents`` does.
"""

# Key on a per-candidate data dict carrying the mode retrieval actually ran in.
# Written once per batch, on the dict the Retriever was handed.
EFFECTIVE_RETRIEVAL_KEY = "retrieval_setting_effective"

# Retrieval never ran, so there is no effective mode to report. Deliberately
# distinct from "none", which means retrieval ran and selected nothing on
# purpose. The manifest is assembled from a ``finally`` block, so it is written
# even for a batch that died before the Retriever was reached; that case must
# say so out loud rather than serialize a null the reader has to interpret.
RETRIEVAL_NOT_ATTEMPTED = "not_attempted"
