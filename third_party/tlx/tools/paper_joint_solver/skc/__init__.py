"""SKC — Schedule-directed Kernel Compiler (Phase A, TLX backend).

Instantiates verified TLX kernel skeletons from joint-solver solutions:
RoleClassifier maps solver warp groups onto skeleton roles, ScheduleBinder
extracts the bindable parameter subset (group count, ring depths, issue
order, phase interleave, register quotas), SkeletonInstantiator renders a
runnable kernel module with a full binding-audit header.
"""
