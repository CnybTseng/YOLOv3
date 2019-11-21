import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--num-classes', type=int, default=20, help='number of classes')
    parser.add_argument('--num-images', type=int, default=5011, help='number of images for training')
    parser.add_argument('--ipc', type=int, default=2000, help='iterations per class')
    parser.add_argument('--milestones', type=str, default='0.8,0.9', help='list of ratio of batch total, must be increasing')
    args = parser.parse_args()
    print(args)
    
    batch_size = args.batch_size
    num_classes = args.num_classes
    num_images = args.num_images
    iterations_per_class = args.ipc
    num_iterations = num_classes * iterations_per_class

    milestones = [float(ms) for ms in args.milestones.split(',')]
    step1 = int(num_iterations * milestones[0])
    step2 = int(num_iterations * milestones[1])

    num_epochs = int(num_iterations * batch_size / num_images)

    print(f'total batches: {num_iterations}')
    print(f'milestones: {step1}, {step2}')
    print(f'num_epochs: {num_epochs}')