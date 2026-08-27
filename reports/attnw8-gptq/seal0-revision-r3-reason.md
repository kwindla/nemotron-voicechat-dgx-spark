# Seal-0 revision 3 reason

Seal-0 revision 2 baseline evidence is invalidated in full. Its first A2
scenario emitted the expected exact tool name and empty arguments with no
fatal or error, and the capture wrapper produced replay payloads. The RNNT
transcript was word-for-word correct but, as is normal for this event stream,
omitted terminal punctuation. Revision 2's generated expected-transcript
field had artificially re-added a period, so the sealed A0 exact-string check
would reject every otherwise exact scenario.

No Hessian, quantized weight, candidate artifact/output, threshold change, or
qualification measurement was produced. The baseline runner was interrupted
as soon as the mismatch was identified, and fence-r2 was restored before this
revision.

Revision 3 replaces every complete tool fixture record with a newly generated
manifest whose expected transcript is the exact normalized RNNT word string,
without synthetic terminal punctuation. Prompts, audio bytes/hashes, scenario
IDs, tool payloads, outcomes, partitions, and every other rule remain fixed.
The entire baseline-only A0/index stage restarts in new directories, and no
revision-2 evidence may enter Seal-1.
