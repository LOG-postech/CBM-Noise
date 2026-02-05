"""
Evaluate trained models on the official CUB test set
"""
import os
import sys
import torch
import joblib
import argparse
import pdb
import numpy as np
from tqdm import tqdm
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import homogeneity_score, f1_score 

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from CUB.dataset import load_data
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES
from analysis import AverageMeter, multiclass_metric, accuracy, binary_accuracy
from CUB.template_model import inception_v3, get_backbone

K = [1, 3, 5] #top k class accuracies to compute

def eval(args):
    """
    Run inference using model (and model2 if bottleneck)
    Returns: (for notebook analysis)
    all_class_labels: flattened list of class labels for each image
    topk_class_outputs: array of top k class ids predicted for each image. Shape = size of test set * max(K)
    all_class_outputs: array of all logit outputs for class prediction, shape = N_TEST * N_CLASS
    all_attr_labels: flattened list of labels for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    all_attr_outputs: flatted list of attribute logits (after ReLU/ Sigmoid respectively) predicted for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    all_attr_outputs_sigmoid: flatted list of attribute logits predicted (after Sigmoid) for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    wrong_idx: image ids where the model got the wrong class prediction (to compare with other models)
    """
    if args.model_dir:
        model = torch.load(args.model_dir)
    else:
        model = None

    if not hasattr(model, 'use_relu'):
        if args.use_relu:
            model.use_relu = True
        else:
            model.use_relu = False
    if not hasattr(model, 'use_sigmoid'):
        if args.use_sigmoid:
            model.use_sigmoid = True
        else:
            model.use_sigmoid = False
    if not hasattr(model, 'cy_fc'):
        model.cy_fc = None
    model.eval()

    if args.model_dir2:
        if 'rf' in args.model_dir2:
            model2 = joblib.load(args.model_dir2)
        else:
            model2 = torch.load(args.model_dir2)
        if not hasattr(model2, 'use_relu'):
            if args.use_relu:
                model2.use_relu = True
            else:
                model2.use_relu = False
        if not hasattr(model2, 'use_sigmoid'):
            if args.use_sigmoid:
                model2.use_sigmoid = True
            else:
                model2.use_sigmoid = False
        model2.eval()
    else:
        model2 = None

    if args.use_attr:
        attr_acc_meter = [AverageMeter()]
        if args.feature_group_results:  # compute acc for each feature individually in addition to the overall accuracy
            for _ in range(args.n_attributes):
                attr_acc_meter.append(AverageMeter())
    else:
        attr_acc_meter = None

    class_acc_meter = []
    for j in range(len(K)):
        class_acc_meter.append(AverageMeter())

    if args.dataset == 'cub':
        data_dir = os.path.join(BASE_DIR, args.data_dir, args.eval_data + '.pkl')
        loader = load_data([data_dir], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir,
                        n_class_attr=args.n_class_attr, args=args)
    elif args.dataset == 'awa2':
        from AWA2.awa2 import generate_data
        TEST_CSV = os.path.join(args.data_dir, 'test.csv')
        _,_,loader,_,_= generate_data(args, args.predicate_dir, "", "", TEST_CSV, resol=299)
        print(len(loader.dataset), "test set size")

    all_outputs, all_targets = [], []
    all_attr_labels, all_attr_outputs, all_attr_outputs_sigmoid, all_attr_outputs2 = [], [], [], []
    all_class_labels, all_class_outputs, all_class_logits = [], [], []
    topk_class_labels, topk_class_outputs = [], []
    report = ""
    c_vec, c_test, y_test = [], [], []

    for data_idx, data in enumerate(loader):
        if args.use_attr:
            if args.no_img:  # A -> Y
                inputs, labels = data
                if isinstance(inputs, list):
                    inputs = torch.stack(inputs).t().float()
                inputs = inputs.float()
            else:
                inputs, labels, attr_labels = data
                if args.dataset == 'cub':
                    attr_labels = torch.stack(attr_labels).t()  # N x 312
        else:  # simple finetune
            inputs, labels = data

        inputs_var = torch.autograd.Variable(inputs).cuda()
        labels_var = torch.autograd.Variable(labels).cuda()
        
        if args.attribute_group:
            outputs = []
            f = open(args.attribute_group, 'r')
            for line in f:
                attr_model = torch.load(line.strip())
                outputs.extend(attr_model(inputs_var))
        else:
            outputs = model(inputs_var)
        if args.use_attr:
            if args.no_img:  # A -> Y
                class_outputs = outputs
            else:
                if args.bottleneck:
                    if args.use_relu:
                        attr_outputs = [torch.nn.ReLU()(o) for o in outputs]
                        attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs]
                    elif args.use_sigmoid:
                        attr_outputs = [torch.nn.Sigmoid()(o) for o in outputs]
                        attr_outputs_sigmoid = attr_outputs
                    else:
                        attr_outputs = outputs
                        attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs]
                    if model2:
                        stage2_inputs = torch.cat(attr_outputs, dim=1)
                        class_outputs = model2(stage2_inputs)
                    else:  # for debugging bottleneck performance without running stage 2
                        class_outputs = torch.zeros([inputs.size(0), N_CLASSES],
                                                    dtype=torch.float64).cuda()  # ignore this
                else:  # cotraining, end2end
                    if args.use_relu:
                        attr_outputs = [torch.nn.ReLU()(o) for o in outputs[1:]]
                        attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs[1:]]
                    elif args.use_sigmoid:
                        attr_outputs = [torch.nn.Sigmoid()(o) for o in outputs[1:]]
                        attr_outputs_sigmoid = attr_outputs
                    else:
                        attr_outputs = outputs[1:]
                        attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs[1:]]

                    class_outputs = outputs[0]

                for i in range(args.n_attributes):
                    acc = binary_accuracy(attr_outputs_sigmoid[i].squeeze(), attr_labels[:, i])
                    acc = acc.data.cpu().numpy()
                    attr_acc_meter[0].update(acc, inputs.size(0))
                    if args.feature_group_results:  # keep track of accuracy of individual attributes
                        attr_acc_meter[i + 1].update(acc, inputs.size(0))

                attr_outputs = torch.cat([o.unsqueeze(1) for o in attr_outputs], dim=1)
                attr_outputs_sigmoid = torch.cat([o for o in attr_outputs_sigmoid], dim=1)
                all_attr_outputs.extend(list(attr_outputs.flatten().data.cpu().numpy()))
                all_attr_outputs_sigmoid.extend(list(attr_outputs_sigmoid.flatten().data.cpu().numpy()))
                all_attr_labels.extend(list(attr_labels.flatten().data.cpu().numpy()))
                c_test.append(attr_labels.data.cpu().numpy())
                c_vec.append(attr_outputs_sigmoid.data.cpu().numpy())
        else:
            class_outputs = outputs[0]
        _, topk_preds = class_outputs.topk(max(K), 1, True, True)
        _, preds = class_outputs.topk(1, 1, True, True)
        all_class_outputs.extend(list(preds.detach().cpu().numpy().flatten()))
        all_class_labels.extend(list(labels.data.cpu().numpy()))
        all_class_logits.extend(class_outputs.detach().cpu().numpy())
        topk_class_outputs.extend(topk_preds.detach().cpu().numpy())
        topk_class_labels.extend(labels.reshape(-1, 1).expand_as(preds))
        y_test.append(labels.data.cpu().numpy())

        np.set_printoptions(threshold=sys.maxsize)

        class_acc = accuracy(class_outputs, labels, topk=K)  # only class prediction accuracy
        for m in range(len(class_acc_meter)):
            class_acc_meter[m].update(class_acc[m], inputs.size(0))

    all_class_logits = np.vstack(all_class_logits)
    topk_class_outputs = np.vstack(topk_class_outputs)
    topk_class_labels = np.vstack(topk_class_labels)
    wrong_idx = np.where(np.sum(topk_class_outputs == topk_class_labels, axis=1) == 0)[0]
    c_test = np.concatenate(c_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    c_vec = np.concatenate(c_vec, axis=0)

    for j in range(len(K)):
        print('Average top %d class accuracy: %.5f' % (K[j], class_acc_meter[j].avg))

    if args.use_attr and not args.no_img:  # print some metrics for attribute prediction performance
        print('Average attribute accuracy: %.5f' % attr_acc_meter[0].avg)
        all_attr_outputs_int = np.array(all_attr_outputs_sigmoid) >= 0.5

        if args.cas:
            print("Computing Concept Alignment Score (CAS)...")
            cas_score, task_alignment_score = concept_alignment_score(
                c_vec= c_vec,
                c_test=c_test,
                y_test=y_test,
                step=50,
                force_alignment=False,
                progress_bar=True,
                bar_idx=args.bar_idx
            )
            print(f"CAS Score: {cas_score:.4f}, Task Alignment Score: {task_alignment_score:.4f}")
            
        if args.feature_group_results:
            n = len(all_attr_labels)
            all_attr_acc, all_attr_f1 = [], []
            for i in range(args.n_attributes):
                acc_meter = attr_acc_meter[1 + i]
                attr_acc = float(acc_meter.avg)
                attr_preds = [all_attr_outputs_int[j] for j in range(n) if j % args.n_attributes == i]
                attr_labels = [all_attr_labels[j] for j in range(n) if j % args.n_attributes == i]
                attr_f1 = f1_score(attr_labels, attr_preds)
                all_attr_acc.append(attr_acc)
                all_attr_f1.append(attr_f1)

            '''
            fig, axs = plt.subplots(1, 2, figsize=(20,10))
            for plt_id, values in enumerate([all_attr_acc, all_attr_f1]):
                axs[plt_id].set_xticks(np.arange(0, 1.1, 0.1))
                if plt_id == 0:
                    axs[plt_id].hist(np.array(values)/100.0, bins=np.arange(0, 1.1, 0.1), rwidth=0.8)
                    axs[plt_id].set_title("Attribute accuracies distribution")
                else:
                    axs[plt_id].hist(values, bins=np.arange(0, 1.1, 0.1), rwidth=0.8)
                    axs[plt_id].set_title("Attribute F1 scores distribution")
            plt.savefig('/'.join(args.model_dir.split('/')[:-1]) + '.png')
            '''
            bins = np.arange(0, 1.01, 0.1)
            acc_bin_ids = np.digitize(np.array(all_attr_acc) / 100.0, bins)
            acc_counts_per_bin = [np.sum(acc_bin_ids == (i + 1)) for i in range(len(bins))]
            f1_bin_ids = np.digitize(np.array(all_attr_f1), bins)
            f1_counts_per_bin = [np.sum(f1_bin_ids == (i + 1)) for i in range(len(bins))]
            print("Accuracy bins:")
            print(acc_counts_per_bin)
            print("F1 bins:")
            print(f1_counts_per_bin)
            np.savetxt(os.path.join(args.log_dir, 'concepts.txt'), f1_counts_per_bin)

        balanced_acc, report = multiclass_metric(all_attr_outputs_int, all_attr_labels)
        f1 = f1_score(all_attr_labels, all_attr_outputs_int)
        print("Total 1's predicted:", sum(np.array(all_attr_outputs_sigmoid) >= 0.5) / len(all_attr_outputs_sigmoid))
        print('Avg attribute balanced acc: %.5f' % (balanced_acc))
        print("Avg attribute F1 score: %.5f" % f1)
        print(report + '\n')

    return class_acc_meter, attr_acc_meter, report, all_class_labels, topk_class_outputs, all_class_logits, all_attr_labels, all_attr_outputs, all_attr_outputs_sigmoid, wrong_idx, all_attr_outputs2

def concept_alignment_score(
    c_vec,
    c_test,
    y_test,
    step,
    force_alignment=False,
    alignment=None,
    progress_bar=True,
    bar_idx = 0
):
    """
    Computes the concept alignment score between learnt concepts and labels.

    :param c_vec: predicted concept representations (can be concept embeddings)
    :param c_test: concept ground truth labels
    :param y_test: task ground truth labels
    :param step: number of integration steps
    :return: concept alignment AUC, task alignment AUC
    """

    # First lets compute an alignment between concept
    # scores and ground truth concepts
    if force_alignment:
        if alignment is None:
            purity_mat = purity.concept_purity_matrix(
                c_soft=c_vec,
                c_true=c_test,
            )
            alignment = purity.find_max_alignment(purity_mat)
        # And use the new vector with its corresponding alignment
        if c_vec.shape[-1] < c_test.shape[-1]:
            # Then the alignment will need to be done backwards as
            # we will have to get rid of the dimensions in c_test
            # which have no aligment at all
            c_test = c_test[:, list(filter(lambda x: x is not None, alignment))]
        else:
            c_vec = c_vec[:, alignment]

    # compute the maximum value for the AUC
    n_clusters = np.linspace(
        2,
        c_vec.shape[0],
        step,
    ).astype(int)
    # print("in cas c_vec shape is", c_vec.shape)
    # print("n_clusters is", n_clusters)
    # print("step is", step)
    max_auc = np.trapz(np.ones(len(n_clusters)))

    concept_auc, task_auc = [], []
    bar = range(c_test.shape[1])
    # bar의 특정 index 부터 시작
    for concept_id in bar[bar_idx:]:
        print("concept_id is", concept_id)
        concept_homogeneity, task_homogeneity = [], []
        for nc in n_clusters:
            print("nc is", nc)
            kmedoids = KMedoids(n_clusters=nc, random_state=0)
            if c_vec.shape[1] != c_test.shape[1]:
                c_cluster_labels = kmedoids.fit_predict(
                    np.hstack([
                        c_vec[:, concept_id][:, np.newaxis],
                        c_vec[:, c_test.shape[1]:]
                    ])
                )
            elif c_vec.shape[1] == c_test.shape[1] and len(c_vec.shape) == 2:
                c_cluster_labels = kmedoids.fit_predict(
                    c_vec[:, concept_id].reshape(-1, 1)
                )
            else:
                c_cluster_labels = kmedoids.fit_predict(c_vec[:, concept_id, :])

            # compute alignment with ground truth labels
            concept_homogeneity.append(
                homogeneity_score(c_test[:, concept_id], c_cluster_labels)
            )
            task_homogeneity.append(
                homogeneity_score(y_test, c_cluster_labels)
            )

        # compute the area under the curve
        concept_auc.append(np.trapz(np.array(concept_homogeneity)) / max_auc)
        task_auc.append(np.trapz(np.array(task_homogeneity)) / max_auc)

    # return the average alignment across all concepts
    concept_auc = np.mean(concept_auc)
    task_auc = np.mean(task_auc)
    if force_alignment:
        return concept_auc, task_auc, alignment
    return concept_auc, task_auc

if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-dataset', default='cub', help='dataset to use')
    parser.add_argument('-log_dir', default='.', help='where results are stored')
    parser.add_argument('-model_dirs', default=None, nargs='+', help='where the trained models are saved')
    parser.add_argument('-model_dirs2', default=None, nargs='+', help='where another trained model are saved (for bottleneck only)')
    parser.add_argument('-eval_data', default='test', help='Type of data (train/ val/ test) to be used')
    parser.add_argument('-use_attr', help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)', action='store_true')
    parser.add_argument('-no_img', help='if included, only use attributes (and not raw imgs) for class prediction', action='store_true')
    parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-n_class_attr', type=int, default=2, help='whether attr prediction is a binary or triary classification')
    parser.add_argument('-data_dir', default='', help='directory to the data used for evaluation')
    parser.add_argument('-n_classes', type=int, default=N_CLASSES, help='number of classes')
    parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES, help='whether to apply bottlenecks to only a few attributes')    
    parser.add_argument('-attribute_group', default=None, help='file listing the (trained) model directory for each attribute group')
    parser.add_argument('-feature_group_results', help='whether to print out performance of individual atttributes', action='store_true')
    parser.add_argument('-use_relu', help='Whether to include relu activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-use_sigmoid', help='Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-rho', type=float, default=0.0, help='rho value for noise injection')
    parser.add_argument('-noise', type=float, default=0.0, help='noise level for noise injection')
    parser.add_argument('-optimizer', default='SGD', help='optimizer')
    parser.add_argument('-noise_loc', type=str, default='concept', help='noise location')
    parser.add_argument('-backbone', type=str, default='inception_v3', help='backbone model')
    parser.add_argument('-expand_dim', type=int, default=0, help='expand dim for MLP')
    parser.add_argument('-cbm_type', type=str, default='independent', help='cbm type')
    parser.add_argument('-predicate_dir', type=str, default='predicate-matrix-continuous.txt', help='predicate matrix')
    parser.add_argument('-cas', help='need cas', action='store_true')
    parser.add_argument('-bar_idx', type=int, default=0, help='bar index')
    args = parser.parse_args()
    args.batch_size = 32

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        
    print(args)
    y_results, c_results = [], []
    for i, model_dir in enumerate(args.model_dirs):
        args.model_dir = model_dir
        args.model_dir2 = args.model_dirs2[i] if args.model_dirs2 else None
        result = eval(args)
        class_acc_meter, attr_acc_meter, report = result[0], result[1], result[2]
        y_results.append(class_acc_meter[0].avg[0].item())
        if attr_acc_meter is not None and not args.no_img:
            c_results.append(attr_acc_meter[0].avg.item())
        else:
            c_results.append(-100.)
    values = (np.mean(c_results), np.std(c_results), np.mean(y_results), np.std(y_results))
    output_string = 'c_mean : %.4f c_std : %.4f y_mean : %.4f y_std : %.4f' % values
    print_string = 'Inference: Acc of C: %.4f +- %.4f, Acc of y: %.4f +- %.4f' % values
    print(output_string)
    print(print_string)

    # make args.log_dir
    output = open(os.path.join(args.log_dir, 'results.txt'), 'w')
    output.write(output_string)
    output.write("\n")
    output.write(print_string)
    output.write("\n")
    output.write(report)