// A failed sibling GPU suite must not hide this job's last successful nightly.
module.exports = async ({github, context, jobName, platform}) => {
  const repo = {owner : context.repo.owner, repo : context.repo.repo};
  const {data : current} = await github.rest.actions.getWorkflowRun({
    ...repo,
    run_id : context.runId,
  });
  const cutoff = new Date(current.created_at);
  const gpu = [ "h100", "b200", "mi350" ].includes(platform);
  // Keep pre-migration history available until the new parent has green runs:
  // the per-GPU workflows', and gpu-tests.yml's for suites that moved to
  // gpu-tests-nightly.yml.
  const workflows =
      gpu ? [ current.workflow_id, "gpu-tests.yml", `${platform}.yml` ]
          : [ current.workflow_id ];
  const candidates = [];
  for (const workflow_id of workflows) {
    const {data} = await github.rest.actions.listWorkflowRuns({
      ...repo,
      workflow_id,
      branch : current.head_branch,
      event : current.event,
      status : gpu ? "completed" : "success",
      per_page : 30,
    });
    candidates.push(...data.workflow_runs);
  }
  candidates.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
  for (const run of candidates) {
    if (run.id === context.runId || new Date(run.created_at) >= cutoff)
      continue;
    if (!gpu)
      return run.head_sha; // Preserve the compiler workflow's behavior.
    const jobs =
        await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
          ...repo,
          run_id : run.id,
          filter : "latest",
          per_page : 100,
        });
    // Called per GPU as "h100 / <job>" (gpu-tests.yml) or as a matrix value,
    // "tritonbench (h100) / <job>" (gpu-tests-nightly.yml).
    const matching = jobs.filter(job => job.name === jobName ||
                                        job.name.endsWith(` / ${jobName}`));
    if (matching.length &&
        matching.every(job => job.conclusion === "success")) {
      return run.head_sha;
    }
  }
  return "";
};
