-- Stakeholder summary metrics for D05
SELECT
  COUNT(*) AS total_records,
  AVG(CAST(on_time AS FLOAT)) AS on_time_rate
FROM project_delivery;

SELECT *
FROM project_delivery
ORDER BY 1
LIMIT 10;
