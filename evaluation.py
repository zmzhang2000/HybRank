import logging
from tqdm import tqdm
import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from models import compute_loss
from utils import Metrics

logger = logging.getLogger()


def eval_datasets(args, dev_loader, test_loader, model=None):
    if dev_loader:
        logger.info(f"{'*' * 10} Evaluating on dev set {'*' * 10}")
        dev_metric = evaluate(args, dev_loader, model)
    else:
        dev_metric = None
    
    if test_loader:
        logger.info(f"{'*' * 10} Evaluating on test set {'*' * 10}")
        test_metric = evaluate(args, test_loader, model)
    else:
        test_metric = None
    
    return dev_metric, test_metric


@torch.no_grad()
def evaluate(args, dataloader, model=None):
    if model:
        model.eval()
    metrics = Metrics()
    dataloader.dataset.data['pytrec_results'] = dict()
    dataloader.dataset.data['pytrec_qrels'] = dict()
    for idx, (qids, sim_matrix, targets) in tqdm(enumerate(dataloader), desc="Evaluating"):
        sim_matrix, targets = sim_matrix.to(args.device), targets.to(args.device)

        if model:
            scores = model(sim_matrix)
            loss = compute_loss(scores, targets)
            metrics.update(loss=(loss.item(), len(sim_matrix)))
        else:
            scores = torch.arange(sim_matrix.shape[-1], 0, -1, 
                                  device=sim_matrix.device)[None, :].expand(sim_matrix.shape[0], -1)

        for qid, score_list in zip(qids, scores):
            qdata = dataloader.dataset.data[qid]
            
            pytrec_results = {pid:score.item() for pid, score in zip(qdata['retrieved_ctxs'], score_list)}
            dataloader.dataset.data['pytrec_results'][qid] = pytrec_results
            
            pytrec_qrels = {pid:1 for pid in qdata['positive_ctxs']}
            if 'has_answer' in qdata:
                pytrec_qrels.update({pid:1 for pid, has_answer in zip(qdata['retrieved_ctxs'], qdata['has_answer']) if has_answer})
            dataloader.dataset.data['pytrec_qrels'][qid] = pytrec_qrels

    if model:
        logger.info("loss: " + str(metrics.meters['loss']))
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(dataloader.dataset.data['pytrec_qrels'], dataloader.dataset.data['pytrec_results'], [1, 5, 10, 20, 50, 100], ignore_identical_ids=False)
    mrr = EvaluateRetrieval.evaluate_custom(dataloader.dataset.data['pytrec_qrels'], dataloader.dataset.data['pytrec_results'], [1, 5, 10, 20, 50, 100], metric="mrr@k")
    accuracy = EvaluateRetrieval.evaluate_custom(dataloader.dataset.data['pytrec_qrels'], dataloader.dataset.data['pytrec_results'], [1, 5, 10, 20, 50, 100], metric="accuracy@k")
    metrics.update(**ndcg, **mrr, **accuracy)
    logger.info("\n")
    return metrics
