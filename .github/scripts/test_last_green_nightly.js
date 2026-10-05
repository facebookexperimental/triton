const assert = require('node:assert/strict');
const test = require('node:test');
const findLastGreen = require('./last_green_nightly.js');

function fixture(runs, jobs = {}) {
  const requests = [];
  const github = {
    rest : {
      actions : {
        getWorkflowRun : async () => ({
          data : {
            workflow_id : 99,
            created_at : '2026-10-05T06:00:00Z',
          }
        }),
        listWorkflowRuns : async args => {
          requests.push(args);
          return {data : {workflow_runs : runs[args.workflow_id] || []}};
        },
        listJobsForWorkflowRun : 'jobs',
      }
    },
    paginate : async (endpoint, args) => {
      assert.equal(endpoint, 'jobs');
      assert.equal(args.filter, 'latest');
      return jobs[args.run_id] || [];
    },
  };
  const context = {repo : {owner : 'owner', repo : 'repo'}, runId : 100};
  return {github, context, requests};
}

function run(id, created_at, conclusion = 'failure') {
  return {id, created_at, conclusion, head_sha : `sha-${id}`};
}

test('a failed sibling does not hide a successful GPU job', async () => {
  const api = fixture({99 : [ run(1, '2026-10-05T00:00:00Z') ]}, {
    1 : [
      {name : 'b200 / b200-tlx-test', conclusion : 'success'},
      {name : 'h100 / h100-tlx-test', conclusion : 'failure'},
    ],
  });
  assert.equal(await findLastGreen(
                   {...api, jobName : 'b200-tlx-test', platform : 'b200'}),
               'sha-1');
  assert.equal(api.requests[0].status, 'completed');
});

test(
    'skipped, missing, and failed jobs are not green; old workflow history works',
    async () => {
      const api = fixture(
          {
            99 : [
              run(3, '2026-10-05T00:00:00Z'), run(2, '2026-10-04T18:00:00Z')
            ],
            'mi350.yml' : [ run(1, '2026-10-04T12:00:00Z', 'success') ],
          },
          {
            3 : [ {name : 'mi350 / mi350-tlx-test', conclusion : 'skipped'} ],
            2 : [ {name : 'mi350 / mi350-tlx-test', conclusion : 'failure'} ],
            1 : [ {name : 'mi350-tlx-test', conclusion : 'success'} ],
          });
      assert.equal(
          await findLastGreen(
              {...api, jobName : 'mi350-tlx-test', platform : 'mi350'}),
          'sha-1');
      assert.equal(await findLastGreen(
                       {...api, jobName : 'missing', platform : 'mi350'}),
                   '');
    });

test(
    'history is ordered across old and new workflows and ignores current/future runs',
    async () => {
      const api = fixture(
          {
            99 : [
              run(100, '2026-10-05T06:00:00Z'), run(4, '2026-10-05T07:00:00Z'),
              run(1, '2026-10-04T12:00:00Z')
            ],
            'h100.yml' : [ run(2, '2026-10-05T00:00:00Z') ],
          },
          {
            100 : [ {name : 'h100 / h100-tlx-test', conclusion : 'success'} ],
            4 : [ {name : 'h100 / h100-tlx-test', conclusion : 'success'} ],
            2 : [ {name : 'h100-tlx-test', conclusion : 'success'} ],
            1 : [ {name : 'h100 / h100-tlx-test', conclusion : 'success'} ],
          });
      assert.equal(await findLastGreen(
                       {...api, jobName : 'h100-tlx-test', platform : 'h100'}),
                   'sha-2');
    });

test('compiler reporting continues to use successful workflow runs',
     async () => {
       const api =
           fixture({99 : [ run(1, '2026-10-05T00:00:00Z', 'success') ]});
       assert.equal(
           await findLastGreen({...api, jobName : 'lit-tests', platform : ''}),
           'sha-1');
       assert.equal(api.requests.length, 1);
       assert.equal(api.requests[0].status, 'success');
     });
